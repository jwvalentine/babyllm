// HuggingFace Text Embedder using ONNX Runtime
// This class handles converting text into vector embeddings using a pre-trained HuggingFace model
// The embeddings are used for semantic similarity search in the RAG pipeline

using Microsoft.ML.OnnxRuntime;        // ONNX Runtime for model inference
using Microsoft.ML.OnnxRuntime.Tensors; // Tensor operations for model input/output
using System.Text.Json;               // JSON serialization for tokenizer API calls

namespace BabyLLM;

/// <summary>
/// A text embedding service that uses HuggingFace transformer models via ONNX Runtime
/// Converts text strings into dense vector representations for semantic similarity search
/// 
/// Architecture:
/// 1. Text → Tokenizer Service → Token IDs, Attention Masks
/// 2. Token IDs → ONNX Model → Hidden States
/// 3. Hidden States → Mean Pooling → Final Embedding Vector
/// 
/// The class handles both single text embedding and batch processing for efficiency
/// </summary>
public class HuggingFaceEmbedder : IDisposable
{
    // ONNX Runtime session for running the embedding model
    // Handles the neural network inference on tokenized input
    private readonly InferenceSession _session;
    
    // HTTP client for communicating with the external tokenizer service
    // The tokenizer converts raw text into model-compatible token sequences
    private readonly HttpClient _http;
    
    // URL of the tokenizer service endpoint
    // This service handles text preprocessing and tokenization
    private readonly string _tokenizerUrl;

    /// <summary>
    /// Initializes a new instance of the HuggingFaceEmbedder
    /// </summary>
    /// <param name="modelPath">Path to the directory containing the ONNX model file (model.onnx)</param>
    /// <param name="tokenizerUrl">URL of the tokenizer service for text preprocessing</param>
    public HuggingFaceEmbedder(string modelPath, string tokenizerUrl)
    {
        // Load the ONNX model from the specified path
        // The model file should be named "model.onnx" in the given directory
        _session = new InferenceSession(Path.Combine(modelPath, "model.onnx"));
        
        // Initialize HTTP client for tokenizer service communication
        _http = new HttpClient();
        
        // Store the tokenizer service URL for later API calls
        _tokenizerUrl = tokenizerUrl;
    }

    /// <summary>
    /// Converts a single text string into a dense vector embedding
    /// 
    /// Process:
    /// 1. Tokenize the text using the external tokenizer service
    /// 2. Create input tensors for the ONNX model
    /// 3. Run inference through the transformer model
    /// 4. Apply mean pooling to get a fixed-size embedding
    /// </summary>
    /// <param name="text">The input text to convert to an embedding</param>
    /// <returns>A float array representing the text's semantic embedding</returns>
    public async Task<float[]> EmbedAsync(string text)
    {
        // Step 1: Convert text to tokens using the tokenizer service
        var encoding = await TokenizeAsync(text);

        // Step 2: Prepare input tensors for the ONNX model
        // Each tensor has shape [batch_size=1, sequence_length]
        
        // Token IDs: The actual token indices for the text
        var inputIds = new DenseTensor<long>(encoding.Ids, new[] { 1, encoding.Ids.Length });
        
        // Attention Mask: 1 for real tokens, 0 for padding tokens
        // Tells the model which tokens to pay attention to
        var attentionMask = new DenseTensor<long>(encoding.AttentionMask, new[] { 1, encoding.AttentionMask.Length });
        
        // Token Type IDs: Segment information (all 0s for single sentence)
        var tokenTypeIds = new DenseTensor<long>(encoding.TokenTypeIds, new[] { 1, encoding.TokenTypeIds.Length });

        // Step 3: Create named inputs for the ONNX model
        // These names must match the model's expected input names
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputIds),
            NamedOnnxValue.CreateFromTensor("attention_mask", attentionMask),
            NamedOnnxValue.CreateFromTensor("token_type_ids", tokenTypeIds)
        };

        // Run the model inference and get the hidden states
        // The model outputs contextualized embeddings for each token
        using var results = _session.Run(inputs);
        var hidden = results.First().AsEnumerable<float>().ToArray();

        // Step 4: Apply mean pooling to convert variable-length token embeddings
        // into a fixed-size sentence embedding
        
        // Calculate dimensions from the output
        int seqLen = encoding.Ids.Length;        // Number of tokens in the sequence
        int hiddenSize = hidden.Length / seqLen; // Embedding dimension per token

        // Initialize the pooled embedding vector
        var pooled = new float[hiddenSize];
        int validTokens = 0;

        // Sum embeddings for all non-padding tokens (where attention_mask = 1)
        for (int i = 0; i < seqLen; i++)
        {
            // Skip padding tokens (attention_mask = 0)
            if (encoding.AttentionMask[i] == 0) continue;
            
            // Add this token's embedding to the sum
            for (int j = 0; j < hiddenSize; j++)
                pooled[j] += hidden[i * hiddenSize + j];
            validTokens++;
        }

        // Compute the mean by dividing by the number of valid tokens
        for (int j = 0; j < hiddenSize; j++)
            pooled[j] /= Math.Max(1, validTokens);

        return pooled;
    }

    /// <summary>
    /// Converts multiple text strings into embeddings in batch
    /// Currently processes texts sequentially - could be optimized for true batch processing
    /// </summary>
    /// <param name="texts">Collection of text strings to embed</param>
    /// <returns>List of embedding vectors corresponding to each input text</returns>
    public async Task<List<float[]>> EmbedBatchAsync(IEnumerable<string> texts)
    {
        var list = new List<float[]>();
        
        // Process each text individually
        // TODO: This could be optimized to use true batch processing
        // by batching the tokenization and model inference steps
        foreach (var t in texts)
            list.Add(await EmbedAsync(t));
            
        return list;
    }

    /// <summary>
    /// Tokenizes text using an external tokenizer service
    /// Converts raw text into token IDs and attention masks required by the model
    /// </summary>
    /// <param name="text">The raw text to tokenize</param>
    /// <returns>Encoding result containing token IDs, attention mask, and token type IDs</returns>
    private async Task<EncodingResult> TokenizeAsync(string text)
    {
        // Prepare the request body with the text to tokenize
        var body = JsonContent.Create(new { text });
        
        // Call the external tokenizer service
        var r = await _http.PostAsync($"{_tokenizerUrl}/tokenize", body);
        r.EnsureSuccessStatusCode();

        // Parse the tokenizer response
        var json = await r.Content.ReadFromJsonAsync<JsonElement>();
        
        // Extract and convert the tokenization results
        return new EncodingResult
        {
            // Token IDs: Numerical representation of each token
            Ids = json.GetProperty("ids").EnumerateArray().Select(x => (long)x.GetInt32()).ToArray(),
            
            // Attention Mask: 1 for real tokens, 0 for padding
            AttentionMask = json.GetProperty("attention_mask").EnumerateArray().Select(x => (long)x.GetInt32()).ToArray(),
            
            // Token Type IDs: All zeros for single sentence (no sentence pairs)
            TokenTypeIds = Enumerable.Repeat(0L, json.GetProperty("ids").GetArrayLength()).ToArray()
        };
    }

    /// <summary>
    /// Disposes of managed resources (ONNX session and HTTP client)
    /// Implements IDisposable to ensure proper cleanup of unmanaged resources
    /// </summary>
    public void Dispose()
    {
        // Clean up the ONNX Runtime session
        _session.Dispose();
        
        // Clean up the HTTP client
        _http.Dispose();
    }

    /// <summary>
    /// Data transfer object that holds the results of text tokenization
    /// Contains all the inputs needed for the transformer model inference
    /// </summary>
    private class EncodingResult
    {
        /// <summary>
        /// Token IDs: Numerical representation of each token in the vocabulary
        /// These are the primary input to the transformer model
        /// </summary>
        public long[] Ids { get; set; } = Array.Empty<long>();
        
        /// <summary>
        /// Attention Mask: Binary mask indicating which tokens are real vs padding
        /// 1 = real token (pay attention), 0 = padding token (ignore)
        /// </summary>
        public long[] AttentionMask { get; set; } = Array.Empty<long>();
        
        /// <summary>
        /// Token Type IDs: Segment information for distinguishing sentence pairs
        /// All zeros for single sentences, used in models like BERT for sentence pair tasks
        /// </summary>
        public long[] TokenTypeIds { get; set; } = Array.Empty<long>();
    }
}