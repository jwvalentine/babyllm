using NUnit.Framework;
using System.Net.Http;
using System.Net.Http.Json;
using System.Text.Json;
using System.Threading.Tasks;
using System.Linq;

namespace BabyLLM.Tests;

[TestFixture]
public class ApiTests
{
    private TestAppFactory _factory = null!;
    private HttpClient _client = null!;

    [SetUp]
    public void Setup()
    {
        _factory = new TestAppFactory();
        _client = _factory.CreateClient();
    }

    [Test]
    public async Task Ask_ShouldReturnAnswer_WithoutDocker()
    {
        var ask = new { question = "What is BabyLLM?" };
        var response = await _client.PostAsJsonAsync("/api/ask", ask);

        Assert.That(response.IsSuccessStatusCode, Is.True);

        var json = await response.Content.ReadFromJsonAsync<JsonElement>();
        Assert.That(json.GetProperty("answer").GetString(), Is.Not.Null.And.Not.Empty);
    }

    [Test]
    public async Task Ingest_Then_Ask_ShouldReturnSources()
    {
        using var content = new MultipartFormDataContent();
        content.Add(new StringContent("Hello world. This is BabyLLM test doc."), "files", "test.md");

        var ingest = await _client.PostAsync("/api/ingest", content);
        Assert.That(ingest.IsSuccessStatusCode, Is.True);

        var ask = new { question = "What is in the test doc?" };
        var response = await _client.PostAsJsonAsync("/api/ask", ask);

        Assert.That(response.IsSuccessStatusCode, Is.True);

        var json = await response.Content.ReadFromJsonAsync<JsonElement>();
        Assert.That(json.GetProperty("sources").EnumerateArray().Any(), Is.True);
    }
}
