using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.Configuration;

namespace GooseTape.Functions.Host.Tests;

/// <summary>
/// Hosts the function host in memory, configured against the synthetic dataset and a checkpoint
/// directory unique to the test run.
/// </summary>
/// <remarks>Open for subclassing so a test can register extra operations.</remarks>
internal class FunctionHostFactory : WebApplicationFactory<Program>
{
    public const string AccessKey = "test-access-key-not-a-real-secret";
    public const string FunctionNamespace = "goosetape.neural-network";
    public const string FunctionVersion = "1";

    /// <summary>The pixel width of the synthetic dataset the tests train against.</summary>
    public const int SyntheticPixelCount = 20;

    private readonly string _checkpointDirectory =
        Path.Combine(Path.GetTempPath(), $"goosetape-host-tests-{Guid.NewGuid():N}");

    protected override void ConfigureWebHost(IWebHostBuilder builder)
    {
        builder.UseEnvironment("Development");

        builder.ConfigureAppConfiguration((_, configuration) =>
        {
            configuration.AddInMemoryCollection(new Dictionary<string, string?>
            {
                ["Ductape:AccessKey"] = AccessKey,
                ["Ductape:FunctionNamespace"] = FunctionNamespace,
                ["Ductape:FunctionVersion"] = FunctionVersion,
                ["Checkpoints:DirectoryPath"] = _checkpointDirectory,
                ["Dataset:Provider"] = "synthetic",
                ["Dataset:SyntheticTrainingExamplesPerClass"] = "40",
                ["Dataset:SyntheticTestExamplesPerClass"] = "20",
                ["Dataset:SyntheticSeed"] = "4242",
            });
        });
    }

    /// <summary>Builds the well-known route for an operation.</summary>
    public static string RouteFor(string operation) =>
        $"/.well-known/ductape/functions/{FunctionNamespace}/{FunctionVersion}/{operation}";

    /// <summary>Builds the invocation body Ductape would post for an operation.</summary>
    public static string BodyFor(string operation, object input, string invocationId) =>
        JsonSerializer.Serialize(new
        {
            function = new { @namespace = FunctionNamespace, operation, version = FunctionVersion },
            input,
            context = new
            {
                product = "goosetape",
                env = "test",
                feature_id = "feature-test",
                feature_tag = "train-digit-recogniser",
                feature_run_id = "run-test",
                step_tag = operation,
                invocation_id = invocationId,
            },
        });

    /// <summary>Signs a body exactly as the Ductape SDK does.</summary>
    public static string Sign(string timestamp, string body, string key = AccessKey) =>
        Convert.ToHexStringLower(
            HMACSHA256.HashData(Encoding.UTF8.GetBytes(key), Encoding.UTF8.GetBytes($"{timestamp}.{body}")));

    protected override void Dispose(bool disposing)
    {
        base.Dispose(disposing);

        if (disposing && Directory.Exists(_checkpointDirectory))
        {
            Directory.Delete(_checkpointDirectory, recursive: true);
        }
    }
}
