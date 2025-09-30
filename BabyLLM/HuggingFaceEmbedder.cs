using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Text.Json;

namespace BabyLLM;

public class HuggingFaceEmbedder : IDisposable
{
    private readonly InferenceSession _session;
    private readonly HttpClient _http;
    private readonly string _tokenizerUrl;

    public HuggingFaceEmbedder(string modelPath, string tokenizerUrl)
    {
        _session = new InferenceSession(Path.Combine(modelPath, "model.onnx"));
        _http = new HttpClient();
        _tokenizerUrl = tokenizerUrl;
    }

    public async Task<float[]> EmbedAsync(string text)
    {
        var encoding = await TokenizeAsync(text);

        var inputIds = new DenseTensor<long>(encoding.Ids, new[] { 1, encoding.Ids.Length });
        var attentionMask = new DenseTensor<long>(encoding.AttentionMask, new[] { 1, encoding.AttentionMask.Length });
        var tokenTypeIds = new DenseTensor<long>(encoding.TokenTypeIds, new[] { 1, encoding.TokenTypeIds.Length });

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids", inputIds),
            NamedOnnxValue.CreateFromTensor("attention_mask", attentionMask),
            NamedOnnxValue.CreateFromTensor("token_type_ids", tokenTypeIds)
        };

        using var results = _session.Run(inputs);
        var hidden = results.First().AsEnumerable<float>().ToArray();

        int seqLen = encoding.Ids.Length;
        int hiddenSize = hidden.Length / seqLen;

        var pooled = new float[hiddenSize];
        int validTokens = 0;

        for (int i = 0; i < seqLen; i++)
        {
            if (encoding.AttentionMask[i] == 0) continue;
            for (int j = 0; j < hiddenSize; j++)
                pooled[j] += hidden[i * hiddenSize + j];
            validTokens++;
        }

        for (int j = 0; j < hiddenSize; j++)
            pooled[j] /= Math.Max(1, validTokens);

        return pooled;
    }

    public async Task<List<float[]>> EmbedBatchAsync(IEnumerable<string> texts)
    {
        var list = new List<float[]>();
        foreach (var t in texts)
            list.Add(await EmbedAsync(t));
        return list;
    }

    private async Task<EncodingResult> TokenizeAsync(string text)
    {
        var body = JsonContent.Create(new { text });
        var r = await _http.PostAsync($"{_tokenizerUrl}/tokenize", body);
        r.EnsureSuccessStatusCode();

        var json = await r.Content.ReadFromJsonAsync<JsonElement>();
        return new EncodingResult
        {
            Ids = json.GetProperty("ids").EnumerateArray().Select(x => (long)x.GetInt32()).ToArray(),
            AttentionMask = json.GetProperty("attention_mask").EnumerateArray().Select(x => (long)x.GetInt32()).ToArray(),
            TokenTypeIds = Enumerable.Repeat(0L, json.GetProperty("ids").GetArrayLength()).ToArray()
        };
    }

    public void Dispose()
    {
        _session.Dispose();
        _http.Dispose();
    }

    private class EncodingResult
    {
        public long[] Ids { get; set; } = Array.Empty<long>();
        public long[] AttentionMask { get; set; } = Array.Empty<long>();
        public long[] TokenTypeIds { get; set; } = Array.Empty<long>();
    }
}