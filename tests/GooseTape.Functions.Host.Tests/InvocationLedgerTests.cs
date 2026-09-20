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
/// Covers single-use invocation handling: the ledger itself, and a replay through the endpoint.
/// </summary>
public sealed class InvocationLedgerTests
{
    [Fact]
    public async Task AnInvocationIdentifier_RunsOnceAndThenReplays()
    {
        var ledger = InvocationLedger.Create(new ManualClock());
        var executions = 0;

        Task<RecordedInvocation> Execute()
        {
            executions++;
            return Task.FromResult(new RecordedInvocation(200, new { value = executions }));
        }

        var first = await ledger.ExecuteOnceAsync("inv-1", Execute);
        var second = await ledger.ExecuteOnceAsync("inv-1", Execute);

        executions.Should().Be(1);
        second.Should().BeSameAs(first);
        ledger.Contains("inv-1").Should().BeTrue();
    }

    [Fact]
    public async Task DifferentInvocationIdentifiers_EachRun()
    {
        var ledger = InvocationLedger.Create(new ManualClock());
        var executions = 0;

        Task<RecordedInvocation> Execute()
        {
            executions++;
            return Task.FromResult(new RecordedInvocation(200, new { value = executions }));
        }

        await ledger.ExecuteOnceAsync("inv-1", Execute);
        await ledger.ExecuteOnceAsync("inv-2", Execute);

        executions.Should().Be(2);
    }

    [Fact]
    public async Task ConcurrentReplaysOfOneIdentifier_StillExecuteOnce()
    {
        var ledger = InvocationLedger.Create(new ManualClock());
        var executions = 0;
        var release = new TaskCompletionSource();

        async Task<RecordedInvocation> Execute()
        {
            Interlocked.Increment(ref executions);
            await release.Task;
            return new RecordedInvocation(200, new { ok = true });
        }

        var attempts = Enumerable.Range(0, 8).Select(_ => ledger.ExecuteOnceAsync("inv-race", Execute)).ToArray();
        release.SetResult();
        var results = await Task.WhenAll(attempts);

        executions.Should().Be(1);
        results.Should().AllSatisfy(result => result.Should().BeSameAs(results[0]));
    }

    [Fact]
    public async Task OnceTheSignatureWindowHasPassed_TheRecordIsNoLongerReplayable()
    {
        // Nothing is gained by remembering an invocation that can no longer be verified.
        var clock = new ManualClock();
        var ledger = InvocationLedger.Create(clock, TimeSpan.FromMinutes(5));
        await ledger.ExecuteOnceAsync("inv-stale", () => Task.FromResult(new RecordedInvocation(200, new { ok = true })));

        ledger.Contains("inv-stale").Should().BeTrue();
        clock.Advance(TimeSpan.FromMinutes(5).Add(TimeSpan.FromSeconds(1)));

        ledger.Contains("inv-stale").Should().BeFalse();
    }

    [Fact]
    public async Task AFailedInvocation_IsNotRecordedAndCanBeAttemptedAgain()
    {
        var ledger = InvocationLedger.Create(new ManualClock());
        var attempts = 0;

        Task<RecordedInvocation> Execute()
        {
            attempts++;
            return attempts == 1
                ? Task.FromException<RecordedInvocation>(new InvalidOperationException("transient"))
                : Task.FromResult(new RecordedInvocation(200, new { ok = true }));
        }

        var first = async () => await ledger.ExecuteOnceAsync("inv-retry", Execute);
        await first.Should().ThrowAsync<InvalidOperationException>();
        var second = await ledger.ExecuteOnceAsync("inv-retry", Execute);

        attempts.Should().Be(2);
        second.StatusCode.Should().Be(200);
    }

    [Theory]
    [InlineData("")]
    [InlineData("   ")]
    public async Task ABlankIdentifier_IsRejected(string invocationId)
    {
        var ledger = InvocationLedger.Create(new ManualClock());

        var act = async () => await ledger.ExecuteOnceAsync(
            invocationId,
            () => Task.FromResult(new RecordedInvocation(200, new { ok = true })));

        await act.Should().ThrowAsync<ArgumentException>();
    }

    [Fact]
    public async Task ReplayingASignedRequest_DoesNotExecuteTheOperationAgain()
    {
        // The whole point: a captured request is valid for five minutes, so without the ledger it
        // could be re-executed repeatedly inside that window.
        using var factory = new CountingHostFactory();
        var body = FunctionHostFactory.BodyFor(CountingOperation.OperationName, new { }, "inv-replayed");

        var first = await PostAsync(factory, body);
        var second = await PostAsync(factory, body);

        first.Status.Should().Be(HttpStatusCode.OK);
        second.Status.Should().Be(HttpStatusCode.OK);
        second.Body.Should().Be(first.Body);
        factory.Operation.Calls.Should().Be(1);
    }

    [Fact]
    public async Task DistinctSignedRequests_EachExecute()
    {
        using var factory = new CountingHostFactory();

        await PostAsync(factory, FunctionHostFactory.BodyFor(CountingOperation.OperationName, new { }, "inv-a"));
        await PostAsync(factory, FunctionHostFactory.BodyFor(CountingOperation.OperationName, new { }, "inv-b"));

        factory.Operation.Calls.Should().Be(2);
    }

    private static async Task<(HttpStatusCode Status, string Body)> PostAsync(FunctionHostFactory factory, string body)
    {
        var timestamp = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()
            .ToString(System.Globalization.CultureInfo.InvariantCulture);

        var request = new HttpRequestMessage(
            HttpMethod.Post,
            new Uri(FunctionHostFactory.RouteFor(CountingOperation.OperationName), UriKind.Relative))
        {
            Content = new StringContent(body, Encoding.UTF8),
        };

        request.Content.Headers.ContentType = new MediaTypeHeaderValue("application/json");
        request.Headers.Add("x-ductape-timestamp", timestamp);
        request.Headers.Add("x-ductape-signature", FunctionHostFactory.Sign(timestamp, body));
        request.Headers.Add(
            "x-ductape-function",
            $"{FunctionHostFactory.FunctionNamespace}.{CountingOperation.OperationName}@{FunctionHostFactory.FunctionVersion}");

        using var client = factory.CreateClient();
        using var response = await client.SendAsync(request);

        return (response.StatusCode, await response.Content.ReadAsStringAsync());
    }

    /// <summary>A host that also serves an operation counting how often it actually runs.</summary>
    private sealed class CountingHostFactory : FunctionHostFactory
    {
        public CountingOperation Operation { get; } = new();

        protected override void ConfigureWebHost(IWebHostBuilder builder)
        {
            base.ConfigureWebHost(builder);
            builder.ConfigureTestServices(services =>
                services.AddSingleton<IPortableFunctionOperation>(Operation));
        }
    }

    private sealed class CountingOperation : IPortableFunctionOperation
    {
        public const string OperationName = "counting-probe";

        private int _calls;

        public string Name => OperationName;

        public int Calls => Volatile.Read(ref _calls);

        public Task<object> ExecuteAsync(
            JsonElement input,
            InvocationContext context,
            CancellationToken cancellationToken) =>
            Task.FromResult<object>(new { calls = Interlocked.Increment(ref _calls) });
    }

    /// <summary>A clock the test moves by hand, so expiry needs no waiting.</summary>
    private sealed class ManualClock : TimeProvider
    {
        private DateTimeOffset _now = new(2026, 9, 20, 12, 0, 0, TimeSpan.Zero);

        public override DateTimeOffset GetUtcNow() => _now;

        public void Advance(TimeSpan amount) => _now += amount;
    }
}
