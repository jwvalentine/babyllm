// BabyLLM - A simple RAG (Retrieval-Augmented Generation) system
// This application provides REST endpoints for document ingestion and question answering
// using vector embeddings and a local LLM via Ollama

using System.Net.Http.Json;
using System.Reflection;
using System.Text;
using System.Text.Json;
using BabyLLM;

// Configure the web application builder with custom settings
// This sets up the application name, content root, and environment
var builder = WebApplication.CreateBuilder(new WebApplicationOptions
{
    Args = args,
    ContentRootPath = Directory.GetCurrentDirectory(),
    ApplicationName = Assembly.GetExecutingAssembly().GetName().Name,
    EnvironmentName = Environment.GetEnvironmentVariable("ASPNETCORE_ENVIRONMENT") ?? "Production"
});

// ---- Configuration ----
// Load all configuration values with sensible defaults
var config = builder.Configuration;

// External service URLs
string OLLAMA = config["Ollama:Url"] ?? "http://localhost:11434";             // Ollama LLM server
string CHROMA = config["Chroma:Url"] ?? "http://localhost:8000";              // ChromaDB vector database
string COLLECTION = config["Chroma:Collection"] ?? "babyllm_docs";            // ChromaDB collection name
string LLM_MODEL = config["Models:LLM"] ?? "phi3:mini";                       // Ollama model name
string EMBED_MODEL = config["Models:Embed"] ?? "hf";                          // Embedding model identifier
string TOKENIZER_URL = config["Tokenizer:Url"] ?? "http://hf-tokenizer:8082"; // HuggingFace tokenizer service
bool USE_FAKES = config.GetValue<bool>("UseFakes");                           // Enable fake/mock responses for testing
string HF_URL = config["HuggingFace:Url"] ?? "http://localhost:8081";         // HuggingFace embedding service URL

// Initialize the HuggingFace embedder for converting text to vector embeddings
// Uses the all-MiniLM-L6-v2 model which is efficient and good for semantic similarity
var embedder = new HuggingFaceEmbedder(
    modelPath: "models/all-MiniLM-L6-v2",
    tokenizerUrl: TOKENIZER_URL
);

// Configure the configuration pipeline to load settings from multiple sources
// Priority: Environment variables > Environment-specific JSON > Base JSON
builder.Configuration
    .SetBasePath(Directory.GetCurrentDirectory())
    .AddJsonFile("Config/appsettings.json", optional: false, reloadOnChange: true)  // Base configuration
    .AddJsonFile($"Config/appsettings.{builder.Environment.EnvironmentName}.json", optional: true, reloadOnChange: true)  // Environment-specific overrides
    .AddEnvironmentVariables();  // Environment variables take highest priority

// Register services for API documentation
builder.Services.AddEndpointsApiExplorer();  // Enables endpoint discovery for Swagger
builder.Services.AddSwaggerGen();            // Generates OpenAPI/Swagger documentation

// Build the application and configure middleware
var app = builder.Build();
app.UseSwagger();    // Serves the OpenAPI specification
app.UseSwaggerUI();  // Provides the Swagger UI interface

// Create HTTP client for external API calls (Ollama, ChromaDB)
var http = new HttpClient()
{
    Timeout = TimeSpan.FromSeconds(600)
};

// Cache for ChromaDB collection IDs to avoid repeated API calls
// Maps collection name to collection ID for efficient lookups
var CollectionCache = new Dictionary<string, string>();

// ---- ChromaDB Helper Functions ----
// These functions handle communication with the ChromaDB vector database
/// <summary>
/// Ensures that the specified ChromaDB collection exists and caches its ID
/// If the collection doesn't exist, it creates one automatically
/// </summary>
async Task EnsureCollectionAsync()
{
    // Skip collection setup when using fake/mock data
    if (USE_FAKES) return;

    // Return early if collection ID is already cached
    if (CollectionCache.ContainsKey(COLLECTION))
        return;

    // Fetch all existing collections from ChromaDB
    var r = await http.GetAsync($"{CHROMA}/api/v1/collections");
    var json = await r.Content.ReadFromJsonAsync<JsonElement>();

    // Check if our target collection already exists
    if (json.ValueKind == JsonValueKind.Array)
    {
        foreach (var col in json.EnumerateArray())
        {
            if (col.GetProperty("name").GetString() == COLLECTION)
            {
                // Cache the collection ID for future use
                CollectionCache[COLLECTION] = col.GetProperty("id").GetString()!;
                return;
            }
        }
    }
    else if (json.ValueKind == JsonValueKind.Object && json.TryGetProperty("error", out var err))
    {
        Console.WriteLine($"Chroma error while listing collections: {err.GetString()}");
    }

    // Collection not found, so create a new one
    var payload = new { name = COLLECTION };
    var create = await http.PostAsJsonAsync($"{CHROMA}/api/v1/collections", payload);
    var created = await create.Content.ReadFromJsonAsync<JsonElement>();

    // Cache the newly created collection's ID
    if (created.ValueKind == JsonValueKind.Object && created.TryGetProperty("id", out var id))
    {
        CollectionCache[COLLECTION] = id.GetString()!;
    }
    else
    {
        throw new Exception($"Failed to create or retrieve collection {COLLECTION}, response: {created}");
    }
}

/// <summary>
/// Inserts or updates documents in the ChromaDB collection with their embeddings
/// Upsert means it will insert new documents or update existing ones based on ID
/// </summary>
/// <param name="ids">Unique identifiers for each document</param>
/// <param name="docs">The actual document text content</param>
/// <param name="metas">Metadata associated with each document (source file, chunk number, etc.)</param>
/// <param name="embeds">Vector embeddings for each document</param>
async Task UpsertAsync(List<string> ids, List<string> docs, List<Dictionary<string, object>> metas, List<float[]> embeds)
{
    // When using fake data, just add to in-memory collections
    if (USE_FAKES)
    {
        for (int i = 0; i < docs.Count; i++)
        {
            FakeRag.Docs.Add(docs[i]);
            FakeRag.Metas.Add(metas[i]);
        }
        return;
    }

    // Ensure the collection exists before upserting
    await EnsureCollectionAsync();
    var collectionId = CollectionCache[COLLECTION];

    // Send the documents and their embeddings to ChromaDB
    var body = new
    {
    ids,
    documents = docs,
    metadatas = metas,
    embeddings = embeds.Select(e => e.Select(v => (double)v).ToArray()).ToList()
    };

    var res = await http.PostAsJsonAsync($"{CHROMA}/api/v1/collections/{collectionId}/upsert", body);
    var respText = await res.Content.ReadAsStringAsync();
    Console.WriteLine($"Upsert response: {res.StatusCode} - {respText}");
    res.EnsureSuccessStatusCode();
}

/// <summary>
/// Performs semantic search in the ChromaDB collection using vector similarity
/// Returns the most similar documents to the query embedding
/// </summary>
/// <param name="queryEmbedding">Vector embedding of the search query</param>
/// <param name="topK">Number of most similar documents to return (default: 5)</param>
/// <returns>Tuple of document texts and their metadata</returns>
async Task<(List<string> docs, List<Dictionary<string, object>> metas)> QueryAsync(float[] queryEmbedding, int topK = 5)
{
    // When using fake data, return first few documents
    if (USE_FAKES)
    {
        var idxs = Enumerable.Range(0, FakeRag.Docs.Count).Take(topK).ToList();
        var docsFake = idxs.Select(i => FakeRag.Docs[i]).ToList();
        var metasFake = idxs.Select(i => FakeRag.Metas[i]).ToList();
        return (docsFake, metasFake);
    }

    // Ensure collection exists before querying
    await EnsureCollectionAsync();
    var collectionId = CollectionCache[COLLECTION];

    // Prepare the similarity search request
    var body = new
    {
        query_embeddings = new[] { queryEmbedding },  // Vector to search for
        n_results = topK,                             // Number of results to return
        include = new[] { "documents", "metadatas", "distances" }  // What data to include in response
    };

    // Execute the similarity search
    var r = await http.PostAsJsonAsync($"{CHROMA}/api/v1/collections/{collectionId}/query", body);
    var json = await r.Content.ReadFromJsonAsync<JsonElement>();

    // Extract documents and metadata from the response
    var docs = json.GetProperty("documents")[0].EnumerateArray().Select(e => e.GetString() ?? "").ToList();
    var metas = json.GetProperty("metadatas")[0].EnumerateArray().Select(e =>
        JsonSerializer.Deserialize<Dictionary<string, object>>(e.GetRawText())!
    ).ToList();

    return (docs, metas);
}

// ---- Embedding Helper Functions ----
// These functions handle text-to-vector conversion for semantic search
/// <summary>
/// Converts a single text string into a vector embedding for semantic search
/// </summary>
/// <param name="text">The text to convert to an embedding</param>
/// <returns>Vector representation of the text</returns>
Task<float[]> EmbedAsync(string text)
{
    // When using fake data, return simple text statistics as a mock embedding
    if (USE_FAKES)
        return Task.FromResult(new[] {
            (float)text.Length,                    // Character count
            (float)text.Count(char.IsLetter),      // Letter count
            (float)text.Count(char.IsWhiteSpace)   // Whitespace count
        });

    // Use the real HuggingFace embedder for production
    return embedder.EmbedAsync(text);
}

/// <summary>
/// Converts multiple text strings into vector embeddings in a single batch operation
/// More efficient than calling EmbedAsync multiple times for large datasets
/// </summary>
/// <param name="texts">Collection of texts to convert to embeddings</param>
/// <returns>List of vector representations for each input text</returns>
Task<List<float[]>> EmbedBatchAsync(IEnumerable<string> texts)
{
    // When using fake data, return simple text statistics for each text
    if (USE_FAKES)
        return Task.FromResult(
            texts.Select(t => new float[] {
                t.Length,                         // Character count
                t.Count(char.IsLetter),          // Letter count  
                t.Count(char.IsWhiteSpace)       // Whitespace count
            }).ToList()
        );

    // Use the real HuggingFace embedder for batch processing
    return embedder.EmbedBatchAsync(texts);
}

/// <summary>
/// Generates a response using the Ollama LLM based on the provided prompt
/// Uses streaming to handle potentially long responses efficiently
/// </summary>
/// <param name="prompt">The input prompt for the language model</param>
/// <returns>Generated text response from the LLM</returns>
async Task<string> GenerateAsync(string prompt)
{
    // When using fake data, return a truncated version of the prompt
    if (USE_FAKES)
        return "[fake] " + (prompt.Length > 240 ? prompt[..240] + "..." : prompt);

    // Prepare the request for Ollama API with low temperature for focused responses
    var body = new { model = LLM_MODEL, prompt, options = new { temperature = 0.2 } };
    var r = await http.PostAsJsonAsync($"{OLLAMA}/api/generate", body);
    
    // Process the streaming response from Ollama
    var sb = new StringBuilder();
    using var stream = await r.Content.ReadAsStreamAsync();
    using var reader = new StreamReader(stream);
    while (!reader.EndOfStream)
    {
        var line = await reader.ReadLineAsync();
        if (string.IsNullOrWhiteSpace(line)) continue;
        
        // Each line is a JSON object containing a piece of the response
        var je = JsonSerializer.Deserialize<JsonElement>(line);
        if (je.TryGetProperty("response", out var part)) 
            sb.Append(part.GetString());
    }
    
    return sb.ToString().Trim();
}

// ---- Utility Functions ----
// Helper functions for text processing and prompt construction
/// <summary>
/// Splits large text into smaller, overlapping chunks for better embedding and retrieval
/// Overlap ensures important context isn't lost at chunk boundaries
/// </summary>
/// <param name="text">The text to split into chunks</param>
/// <param name="chunkSize">Maximum number of words per chunk (default: 512)</param>
/// <param name="overlap">Number of words to overlap between chunks (default: 64)</param>
/// <returns>List of text chunks with overlap</returns>
static List<string> Chunk(string text, int chunkSize = 512, int overlap = 64)
{
    // Split text into words, removing empty entries
    var words = text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
    var chunks = new List<string>();
    int i = 0;
    
    // Create overlapping chunks by advancing less than the full chunk size
    while (i < words.Length)
    {
        var slice = words.Skip(i).Take(chunkSize);
        chunks.Add(string.Join(' ', slice));
        // Advance by (chunkSize - overlap) to create overlap between chunks
        i += Math.Max(1, chunkSize - overlap);
    }
    
    // Filter out any empty chunks
    return chunks.Where(c => !string.IsNullOrWhiteSpace(c)).ToList();
}

/// <summary>
/// Constructs a RAG (Retrieval-Augmented Generation) prompt by combining
/// retrieved context documents with the user's question
/// </summary>
/// <param name="question">The user's question</param>
/// <param name="ctxs">Retrieved context documents relevant to the question</param>
/// <returns>A formatted prompt for the LLM that includes context and question</returns>
static string BuildPrompt(string question, IEnumerable<string> ctxs)
{
    // Join all context documents with separators for clarity
    var ctx = string.Join("\n\n---\n\n", ctxs);
    
    // Create a structured prompt that instructs the LLM to:
    // 1. Only use the provided context
    // 2. Admit when it doesn't know something
    // 3. Answer the specific question asked
    return
$@"Answer ONLY using the provided context.
If the answer is not in the context, say you don't know.

Context:
{ctx}

Question: {question}
Answer:";
}

// ---- API Endpoints ----
// REST endpoints for document ingestion and question answering
/// <summary>
/// POST /api/ingest - Ingests documents for later retrieval
/// Accepts file uploads, chunks them, creates embeddings, and stores in ChromaDB
/// </summary>
app.MapPost("/api/ingest", async (HttpRequest req) =>
{
    // Ensure the ChromaDB collection exists before processing files
    await EnsureCollectionAsync();

    // Validate that request contains file uploads
    if (!req.HasFormContentType) return Results.BadRequest("multipart/form-data required");
    var form = await req.ReadFormAsync();
    var files = form.Files;
    if (files.Count == 0) return Results.BadRequest("no files");

    // Initialize collections for processed document data
    var ids = new List<string>();       // Unique identifiers for each chunk
    var docs = new List<string>();      // Document text content
    var metas = new List<Dictionary<string, object>>();  // Metadata for each chunk

    // Process each uploaded file
    foreach (var f in files)
    {
        if (f.Length == 0) continue;  // Skip empty files
        
        // Read the entire file content
        using var sr = new StreamReader(f.OpenReadStream());
        var text = await sr.ReadToEndAsync();

        // Basic cleanup for Markdown files - remove formatting characters
        if (f.FileName.EndsWith(".md", StringComparison.OrdinalIgnoreCase))
            text = text.Replace("#", "").Replace("*", "");

        // Split the document into overlapping chunks for better retrieval
        var chunks = Chunk(text);
        
        // Create unique IDs and metadata for each chunk
        for (int i = 0; i < chunks.Count; i++)
        {
            ids.Add($"{f.FileName}:{i}");  // Format: "filename.txt:0", "filename.txt:1", etc.
            docs.Add(chunks[i]);
            metas.Add(new Dictionary<string, object> { 
                ["source"] = f.FileName,  // Original file name
                ["chunk"] = i            // Chunk index within the file
            });
        }
    }

    // Handle fake mode by storing in memory instead of ChromaDB
    if (USE_FAKES)
    {
        for (int i = 0; i < docs.Count; i++)
        {
            FakeRag.Docs.Add(docs[i]);
            FakeRag.Metas.Add(metas[i]);
        }
        return Results.Ok(new { added = docs.Count, fake = true });
    }

    // Process documents in batches to avoid overwhelming the embedding service
    const int batch = 32;  // Process 32 documents at a time
    for (int i = 0; i < docs.Count; i += batch)
    {
        // Create a batch of documents, IDs, and metadata
        var sliceDocs = docs.Skip(i).Take(batch).ToList();
        var sliceIds = ids.Skip(i).Take(batch).ToList();
        var sliceMeta = metas.Skip(i).Take(batch).ToList();
        
        // Generate embeddings for the batch
        var embeds = await EmbedBatchAsync(sliceDocs);
        
        // Store the batch in ChromaDB
        await UpsertAsync(sliceIds, sliceDocs, sliceMeta, embeds);
    }

    return Results.Ok(new { added = docs.Count, fake = false });
})
.WithName("IngestDocs")        // Endpoint name for OpenAPI
.WithOpenApi();               // Include in Swagger documentation

/// <summary>
/// POST /api/ask - Answers questions using RAG (Retrieval-Augmented Generation)
/// 1. Embeds the question
/// 2. Finds similar documents using vector search
/// 3. Constructs a prompt with retrieved context
/// 4. Generates an answer using the LLM
/// </summary>
app.MapPost("/api/ask", async (Ask ask) =>
{
    // Ensure ChromaDB collection exists
    await EnsureCollectionAsync();

    List<string> docs;
    List<Dictionary<string, object>> metas;

    if (USE_FAKES)
    {
        // In fake mode, just return the first few stored documents
        docs = FakeRag.Docs.Take(5).ToList();
        metas = FakeRag.Metas.Take(5).ToList();
    }
    else
    {
        // Real RAG pipeline:
        // 1. Convert question to vector embedding
        var qvec = await EmbedAsync(ask.question);
        // 2. Find the 5 most similar documents in ChromaDB
        (docs, metas) = await QueryAsync(qvec, topK: 5);
    }

    // 3. Construct a prompt that includes the retrieved context
    var prompt = BuildPrompt(ask.question, docs);
    
    // 4. Generate an answer using the LLM with the context
    var answer = await GenerateAsync(prompt);

    // Return both the answer and the source documents for transparency
    return Results.Json(new { answer, sources = metas });
})
.WithName("AskQuestion")      // Endpoint name for OpenAPI
.WithOpenApi();               // Include in Swagger documentation

// Start the web application and listen for requests
app.Run();

// ---- Data Transfer Objects ----
/// <summary>
/// Record for the /api/ask endpoint request body
/// Contains the user's question for the RAG system
/// </summary>
record Ask(string question);
