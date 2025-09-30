// ---- Fake RAG store ----
static class FakeRag
{
    public static readonly List<string> Docs = new();
    public static readonly List<Dictionary<string, object>> Metas = new();
    public static void Clear()
    {
        Docs.Clear();
        Metas.Clear();
    }
}