using System.Net;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using GooseTape.Functions.Host.Ductape;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.DependencyInjection;

namespace GooseTape.Functions.Host.Tests;

/// <summary>
/// Covers the execution boundary's answer to a failure no operation anticipated.
/// </summary>
public sealed class UnexpectedFailureTests
{
    /// <summary>Stands in for internal detail an exception might carry, such as a path or a setting.</summary>
    private const string InternalDetail = @"C:\secrets\connection-string=Server=prod;Password=hunter2";

    [Fact]
    public async Task AnUnanticipatedException_BecomesAStructuredFailure()
    {
        using var factory = new CrashingHostFactory();

        var response = await PostAsync(factory, "inv-crash");

        response.Status.Should().Be(HttpStatusCode.InternalServerError);

        var root = JsonDocument.Parse(response.Body).RootElement;
        root.GetProperty("invocation_id").GetString().Should().Be("inv-crash");

        var error = root.GetProperty("error");
        error.GetProperty("code").GetString().Should().Be("FUNCTION_EXECUTION_FAILED");
        error.GetProperty("retryable").GetBoolean().Should().BeTrue();
        error.GetProperty("message").GetString().Should().Contain("inv-crash");
    }

    [Fact]
    public async Task TheExceptionsOwnDetail_NeverReachesTheCaller()
    {
        using var factory = new CrashingHostFactory();

        var response = await PostAsync(factory, "inv-leak-check");

        response.Body.Should().NotContain("hunter2");
        response.Body.Should().NotContain("secrets");
        response.Body.Should().NotContain(nameof(InvalidOperationException));
    }

    [Fact]
    public async Task AReplayOfACrashedInvocation_GetsTheSameAnswerWithoutRunningAgain()
    {
        // The failure is a recorded outcome like any other, so the ledger replays it.
        using var factory = new CrashingHostFactory();

        var first = await PostAsync(factory, "inv-crash-replay");
        var second = await PostAsync(factory, "inv-crash-replay");

        second.Body.Should().Be(first.Body);
        factory.Operation.Calls.Should().Be(1);
    }

    private static async Task<(HttpStatusCode Status, string Body)> PostAsync(CrashingHostFactory factory, string invocationId)
    {
        var body = FunctionHostFactory.BodyFor(CrashingOperation.OperationName, new { }, invocationId);
        var timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()
            .ToString(System.Globalization.CultureInfo.InvariantCulture);

        var request = new HttpRequestMessage(
            HttpMethod.Post,
            new Uri(FunctionHostFactory.RouteFor(CrashingOperation.OperationName), UriKind.Relative))
        {
            Content = new StringContent(body, Encoding.UTF8),
        };

        request.Content.Headers.ContentType = new MediaTypeHeaderValue("application/json");
        request.Headers.Add("x-ductape-timestamp", timestamp);
        request.Headers.Add("x-ductape-signature", FunctionHostFactory.Sign(timestamp, body));
        request.Headers.Add(
            "x-ductape-function",
            $"{FunctionHostFactory.FunctionNamespace}.{CrashingOperation.OperationName}@{FunctionHostFactory.FunctionVersion}");

        using var client = factory.CreateClient();
        using var response = await client.SendAsync(request);

        return (response.StatusCode, await response.Content.ReadAsStringAsync());
    }

    private sealed class CrashingHostFactory : FunctionHostFactory
    {
        public CrashingOperation Operation { get; } = new();

        protected override void ConfigureWebHost(IWebHostBuilder builder)
        {
            base.ConfigureWebHost(builder);
            builder.ConfigureTestServices(services =>
                services.AddSingleton<IPortableFunctionOperation>(Operation));
        }
    }

    /// <summary>Fails the way a bug would: an exception nobody wrapped, carrying internal detail.</summary>
    private sealed class CrashingOperation : IPortableFunctionOperation
    {
        public const string OperationName = "crashing-probe";

        private int _calls;

        public string Name => OperationName;

        public int Calls => Volatile.Read(ref _calls);

        public Task<object> ExecuteAsync(
            JsonElement input,
            InvocationContext context,
            CancellationToken cancellationToken)
        {
            Interlocked.Increment(ref _calls);
            throw new InvalidOperationException($"Could not open {InternalDetail}");
        }
    }
}
