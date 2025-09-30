#nullable enable
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.Configuration;
using System.Collections.Generic;

namespace BabyLLM.Tests;

public class TestAppFactory : WebApplicationFactory<BabyLLM.Program>
{
    protected override void ConfigureWebHost(IWebHostBuilder builder)
    {
        builder.ConfigureAppConfiguration((context, config) =>
        {
            // Force fake mode during tests so Docker isn't required
            var overrides = new Dictionary<string, string?>
            {
                ["UseFakes"] = "true"
            };
            config.AddInMemoryCollection(overrides!);
        });
    }
}
