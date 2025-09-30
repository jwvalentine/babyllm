using System.Net.Http.Json;
using System.Reflection;
using System.Text;
using System.Text.Json;
using BabyLLM;

var builder = WebApplication.CreateBuilder(new WebApplicationOptions
{
    Args = args,
    ContentRootPath = Directory.GetCurrentDirectory(),
    ApplicationName = Assembly.GetExecutingAssembly().GetName().Name,
    EnvironmentName = Environment.GetEnvironmentVariable("ASPNETCORE_ENVIRONMENT") ?? "Production"
});

// ---- Config ----
var config = builder.Configuration;

string OLLAMA = config["Ollama:Url"] ?? "http://localhost:11434";
string CHROMA = config["Chroma:Url"] ?? "http://localhost:8000";
string COLLECTION = config["Chroma:Collection"] ?? "babyllm_docs";
string LLM_MODEL = config["Models:LLM"] ?? "phi3:mini";
string EMBED_MODEL = config["Models:Embed"] ?? "hf";
string TOKENIZER_URL = config["Tokenizer:Url"] ?? "http://hf-tokenizer:8082";
bool USE_FAKES = config.GetValue<bool>("UseFakes");

var embedder = new HuggingFaceEmbedder(
    modelPath: "models/all-MiniLM-L6-v2",
    tokenizerUrl: TOKENIZER_URL
);

// Load config
builder.Configuration
    .SetBasePath(Directory.GetCurrentDirectory())
    .AddJsonFile("Config/appsettings.json", optional: false, reloadOnChange: true)
    .AddJsonFile($"Config/appsettings.{builder.Environment.EnvironmentName}.json", optional: true, reloadOnChange: true)
    .AddEnvironmentVariables();

builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

var app = builder.Build();
app.UseSwagger();
app.UseSwaggerUI();

var http = new HttpClient();

// Cache for collection IDs → avoids repeat lookups
var CollectionCache = new Dictionary<string, string>();

// ---- Chroma helpers ----
async Task EnsureCollectionAsync()
{
    if (USE_FAKES) return;

    if (CollectionCache.ContainsKey(COLLECTION))
        return;

    var r = await http.GetAsync($"{CHROMA}/api/v1/collections");
    var json = await r.Content.ReadFromJsonAsync<JsonElement>();

    if (json.ValueKind == JsonValueKind.Array)
    {
        foreach (var col in json.EnumerateArray())
        {
            if (col.GetProperty("name").GetString() == COLLECTION)
            {
                CollectionCache[COLLECTION] = col.GetProperty("id").GetString()!;
                return;
            }
        }
    }
    else if (json.ValueKind == JsonValueKind.Object && json.TryGetProperty("error", out var err))
    {
        Console.WriteLine($"Chroma error while listing collections: {err.GetString()}");
    }

    // If not found, create the collection
    var payload = new { name = COLLECTION };
    var create = await http.PostAsJsonAsync($"{CHROMA}/api/v1/collections", payload);
    var created = await create.Content.ReadFromJsonAsync<JsonElement>();

    if (created.ValueKind == JsonValueKind.Object && created.TryGetProperty("id", out var id))
    {
        CollectionCache[COLLECTION] = id.GetString()!;
    }
    else
    {
        throw new Exception($"Failed to create or retrieve collection {COLLECTION}, response: {created}");
    }
}

async Task UpsertAsync(List<string> ids, List<string> docs, List<Dictionary<string, object>> metas, List<float[]> embeds)
{
    if (USE_FAKES)
    {
        for (int i = 0; i < docs.Count; i++)
        {
            FakeRag.Docs.Add(docs[i]);
            FakeRag.Metas.Add(metas[i]);
        }
        return;
    }

    await EnsureCollectionAsync();
    var collectionId = CollectionCache[COLLECTION];

    var body = new { ids, documents = docs, metadatas = metas, embeddings = embeds };
    await http.PostAsJsonAsync($"{CHROMA}/api/v1/collections/{collectionId}/upsert", body);
}

async Task<(List<string> docs, List<Dictionary<string, object>> metas)> QueryAsync(float[] queryEmbedding, int topK = 5)
{
    if (USE_FAKES)
    {
        var idxs = Enumerable.Range(0, FakeRag.Docs.Count).Take(topK).ToList();
        var docsFake = idxs.Select(i => FakeRag.Docs[i]).ToList();
        var metasFake = idxs.Select(i => FakeRag.Metas[i]).ToList();
        return (docsFake, metasFake);
    }

    await EnsureCollectionAsync();
    var collectionId = CollectionCache[COLLECTION];

    var body = new
    {
        query_embeddings = new[] { queryEmbedding },
        n_results = topK,
        include = new[] { "documents", "metadatas", "distances" }
    };

    var r = await http.PostAsJsonAsync($"{CHROMA}/api/v1/collections/{collectionId}/query", body);
    var json = await r.Content.ReadFromJsonAsync<JsonElement>();

    var docs = json.GetProperty("documents")[0].EnumerateArray().Select(e => e.GetString() ?? "").ToList();
    var metas = json.GetProperty("metadatas")[0].EnumerateArray().Select(e =>
        JsonSerializer.Deserialize<Dictionary<string, object>>(e.GetRawText())!
    ).ToList();

    return (docs, metas);
}

string HF_URL = config["HuggingFace:Url"] ?? "http://localhost:8081";

// ---- Embedding helpers ----
Task<float[]> EmbedAsync(string text)
{
    if (USE_FAKES)
        return Task.FromResult(new[] {
            (float)text.Length,
            (float)text.Count(char.IsLetter),
            (float)text.Count(char.IsWhiteSpace)
        });

    return embedder.EmbedAsync(text);
}

Task<List<float[]>> EmbedBatchAsync(IEnumerable<string> texts)
{
    if (USE_FAKES)
        return Task.FromResult(
            texts.Select(t => new float[] {
                t.Length,
                t.Count(char.IsLetter),
                t.Count(char.IsWhiteSpace)
            }).ToList()
        );

    return embedder.EmbedBatchAsync(texts);
}

async Task<string> GenerateAsync(string prompt)
{
    if (USE_FAKES)
        return "[fake] " + (prompt.Length > 240 ? prompt[..240] + "..." : prompt);

    var body = new { model = LLM_MODEL, prompt, options = new { temperature = 0.2 } };
    var r = await http.PostAsJsonAsync($"{OLLAMA}/api/generate", body);
    var sb = new StringBuilder();
    using var stream = await r.Content.ReadAsStreamAsync();
    using var reader = new StreamReader(stream);
    while (!reader.EndOfStream)
    {
        var line = await reader.ReadLineAsync();
        if (string.IsNullOrWhiteSpace(line)) continue;
        var je = JsonSerializer.Deserialize<JsonElement>(line);
        if (je.TryGetProperty("response", out var part)) sb.Append(part.GetString());
    }
    return sb.ToString().Trim();
}

// ---- Utilities ----
static List<string> Chunk(string text, int chunkSize = 800, int overlap = 160)
{
    var words = text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
    var chunks = new List<string>();
    int i = 0;
    while (i < words.Length)
    {
        var slice = words.Skip(i).Take(chunkSize);
        chunks.Add(string.Join(' ', slice));
        i += Math.Max(1, chunkSize - overlap);
    }
    return chunks.Where(c => !string.IsNullOrWhiteSpace(c)).ToList();
}

static string BuildPrompt(string question, IEnumerable<string> ctxs)
{
    var ctx = string.Join("\n\n---\n\n", ctxs);
    return
$@"You are a helpful assistant. Answer ONLY using the provided context.
If the answer is not in the context, say you don't know.

Context:
{ctx}

Question: {question}
Answer:";
}

// ---- Endpoints ----
app.MapPost("/api/ingest", async (HttpRequest req) =>
{
    await EnsureCollectionAsync();

    if (!req.HasFormContentType) return Results.BadRequest("multipart/form-data required");
    var form = await req.ReadFormAsync();
    var files = form.Files;
    if (files.Count == 0) return Results.BadRequest("no files");

    var ids = new List<string>();
    var docs = new List<string>();
    var metas = new List<Dictionary<string, object>>();

    foreach (var f in files)
    {
        if (f.Length == 0) continue;
        using var sr = new StreamReader(f.OpenReadStream());
        var text = await sr.ReadToEndAsync();

        if (f.FileName.EndsWith(".md", StringComparison.OrdinalIgnoreCase))
            text = text.Replace("#", "").Replace("*", "");

        var chunks = Chunk(text);
        for (int i = 0; i < chunks.Count; i++)
        {
            ids.Add($"{f.FileName}:{i}");
            docs.Add(chunks[i]);
            metas.Add(new Dictionary<string, object> { ["source"] = f.FileName, ["chunk"] = i });
        }
    }

    if (USE_FAKES)
    {
        for (int i = 0; i < docs.Count; i++)
        {
            FakeRag.Docs.Add(docs[i]);
            FakeRag.Metas.Add(metas[i]);
        }
        return Results.Ok(new { added = docs.Count, fake = true });
    }

    const int batch = 32;
    for (int i = 0; i < docs.Count; i += batch)
    {
        var sliceDocs = docs.Skip(i).Take(batch).ToList();
        var sliceIds = ids.Skip(i).Take(batch).ToList();
        var sliceMeta = metas.Skip(i).Take(batch).ToList();
        var embeds = await EmbedBatchAsync(sliceDocs);
        await UpsertAsync(sliceIds, sliceDocs, sliceMeta, embeds);
    }

    return Results.Ok(new { added = docs.Count, fake = false });
})
.WithName("IngestDocs")
.WithOpenApi();

app.MapPost("/api/ask", async (Ask ask) =>
{
    await EnsureCollectionAsync();

    List<string> docs;
    List<Dictionary<string, object>> metas;

    if (USE_FAKES)
    {
        docs = FakeRag.Docs.Take(5).ToList();
        metas = FakeRag.Metas.Take(5).ToList();
    }
    else
    {
        var qvec = await EmbedAsync(ask.question);
        (docs, metas) = await QueryAsync(qvec, topK: 5);
    }

    var prompt = BuildPrompt(ask.question, docs);
    var answer = await GenerateAsync(prompt);

    return Results.Json(new { answer, sources = metas });
})
.WithName("AskQuestion")
.WithOpenApi();

app.Run();

record Ask(string question);
