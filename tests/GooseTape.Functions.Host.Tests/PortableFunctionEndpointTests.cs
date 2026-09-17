using System.Net;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;

namespace GooseTape.Functions.Host.Tests;

/// <summary>
/// Exercises the signed HTTP boundary: what it accepts, and what it must refuse.
/// </summary>
public sealed class PortableFunctionEndpointTests : IClassFixture<FunctionHostFactoryFixture>
{
    private static readonly int[] MatchingTopology = [FunctionHostFactory.SyntheticPixelCount, 16, 10];
    private static readonly int[] MismatchedTopology = [999, 16, 10];

    private readonly FunctionHostFactory _factory;

    public PortableFunctionEndpointTests(FunctionHostFactoryFixture fixture)
    {
        ArgumentNullException.ThrowIfNull(fixture);

        _factory = fixture.Factory;
    }

    [Fact]
    public async Task ValidlySignedInvocation_IsAccepted()
    {
        var response = await InvokeAsync("initialize-network", new
        {
            run_id = NewRunId(),
            topology = MatchingTopology,
            seed = 7,
        });

        response.Status.Should().Be(HttpStatusCode.OK);
        Output(response).GetProperty("checkpoint_id").GetString().Should().NotBeNullOrWhiteSpace();
        Output(response).GetProperty("input_size").GetInt32().Should().Be(FunctionHostFactory.SyntheticPixelCount);
        Output(response).GetProperty("layer_count").GetInt32().Should().Be(2);
    }

    [Fact]
    public async Task UnsignedInvocation_IsRejected()
    {
        var body = FunctionHostFactory.BodyFor("evaluate-network", new { checkpoint_id = "anything" }, "inv-unsigned");
        var request = Request("evaluate-network", body, "evaluate-network");

        var response = await SendAsync(request);

        response.Status.Should().Be(HttpStatusCode.Unauthorized);
        ErrorCode(response).Should().Be("FUNCTION_SIGNATURE_INVALID");
    }

    [Fact]
    public async Task TamperedSignature_IsRejected()
    {
        var body = FunctionHostFactory.BodyFor("evaluate-network", new { checkpoint_id = "anything" }, "inv-tampered");
        var request = Request("evaluate-network", body, "evaluate-network");
        request.Headers.Add("x-ductape-timestamp", Now());
        request.Headers.Add("x-ductape-signature", new string('a', 64));

        var response = await SendAsync(request);

        response.Status.Should().Be(HttpStatusCode.Unauthorized);
        ErrorCode(response).Should().Be("FUNCTION_SIGNATURE_INVALID");
    }

    [Fact]
    public async Task SignatureFromTheWrongKey_IsRejected()
    {
        var body = FunctionHostFactory.BodyFor("evaluate-network", new { checkpoint_id = "anything" }, "inv-wrong-key");
        var timestamp = Now();
        var request = Request("evaluate-network", body, "evaluate-network");
        request.Headers.Add("x-ductape-timestamp", timestamp);
        request.Headers.Add("x-ductape-signature", FunctionHostFactory.Sign(timestamp, body, "a-different-key"));

        var response = await SendAsync(request);

        response.Status.Should().Be(HttpStatusCode.Unauthorized);
        ErrorCode(response).Should().Be("FUNCTION_SIGNATURE_INVALID");
    }

    [Fact]
    public async Task ExpiredTimestamp_IsRejectedEvenWhenCorrectlySigned()
    {
        // A capture replayed beyond the five minute window must not be accepted, even though its
        // signature is genuine for that body.
        var body = FunctionHostFactory.BodyFor("evaluate-network", new { checkpoint_id = "anything" }, "inv-stale");
        var staleTimestamp = DateTimeOffset.UtcNow.AddMinutes(-10).ToUnixTimeMilliseconds().ToString(
            System.Globalization.CultureInfo.InvariantCulture);

        var request = Request("evaluate-network", body, "evaluate-network");
        request.Headers.Add("x-ductape-timestamp", staleTimestamp);
        request.Headers.Add("x-ductape-signature", FunctionHostFactory.Sign(staleTimestamp, body));

        var response = await SendAsync(request);

        response.Status.Should().Be(HttpStatusCode.Unauthorized);
        ErrorCode(response).Should().Be("FUNCTION_SIGNATURE_INVALID");
    }

    [Fact]
    public async Task ValidSignatureRedirectedAtAnotherRoute_IsRejected()
    {
        // The body is signed for evaluate-network but posted to the classify-digit route.
        var body = FunctionHostFactory.BodyFor("evaluate-network", new { checkpoint_id = "anything" }, "inv-redirected");
        var timestamp = Now();
        var request = Request("classify-digit", body, "evaluate-network");
        request.Headers.Add("x-ductape-timestamp", timestamp);
        request.Headers.Add("x-ductape-signature", FunctionHostFactory.Sign(timestamp, body));

        var response = await SendAsync(request);

        response.Status.Should().Be(HttpStatusCode.UnprocessableEntity);
        ErrorCode(response).Should().Be("FUNCTION_ROUTE_MISMATCH");
    }

    [Fact]
    public async Task MismatchedFunctionHeader_IsRejected()
    {
        var body = FunctionHostFactory.BodyFor("evaluate-network", new { checkpoint_id = "anything" }, "inv-bad-header");
        var timestamp = Now();
        var request = Request("evaluate-network", body, "classify-digit");
        request.Headers.Add("x-ductape-timestamp", timestamp);
        request.Headers.Add("x-ductape-signature", FunctionHostFactory.Sign(timestamp, body));

        var response = await SendAsync(request);

        response.Status.Should().Be(HttpStatusCode.UnprocessableEntity);
        ErrorCode(response).Should().Be("FUNCTION_ROUTE_MISMATCH");
    }

    [Fact]
    public async Task UnknownOperation_IsReportedAsUnavailable()
    {
        var response = await InvokeAsync("teach-it-to-sing", new { anything = 1 });

        response.Status.Should().Be(HttpStatusCode.ServiceUnavailable);
        ErrorCode(response).Should().Be("FUNCTION_UNAVAILABLE");
    }

    [Fact]
    public async Task MalformedBody_IsRejectedAsInvalidRequest()
    {
        const string body = "{ this is not json";
        var timestamp = Now();
        var request = Request("evaluate-network", body, "evaluate-network");
        request.Headers.Add("x-ductape-timestamp", timestamp);
        request.Headers.Add("x-ductape-signature", FunctionHostFactory.Sign(timestamp, body));

        var response = await SendAsync(request);

        response.Status.Should().Be(HttpStatusCode.BadRequest);
        ErrorCode(response).Should().Be("FUNCTION_REQUEST_INVALID");
    }

    [Fact]
    public async Task MissingRequiredInputField_IsRejectedNamingTheField()
    {
        var response = await InvokeAsync("initialize-network", new { topology = MatchingTopology, seed = 1 });

        response.Status.Should().Be(HttpStatusCode.UnprocessableEntity);
        ErrorCode(response).Should().Be("FUNCTION_INPUT_INVALID");
        ErrorMessage(response).Should().Contain("run_id");
    }

    [Fact]
    public async Task TopologyThatDoesNotMatchTheData_IsRejected()
    {
        var response = await InvokeAsync("initialize-network", new
        {
            run_id = NewRunId(),
            topology = MismatchedTopology,
            seed = 1,
        });

        response.Status.Should().Be(HttpStatusCode.UnprocessableEntity);
        ErrorCode(response).Should().Be("TOPOLOGY_MISMATCH");
    }

    [Theory]
    [InlineData("../../etc/passwd")]
    [InlineData("nested/path")]
    [InlineData("has space")]
    public async Task CheckpointIdentifiersThatCouldEscapeTheStore_AreRejected(string checkpointId)
    {
        var response = await InvokeAsync("evaluate-network", new { checkpoint_id = checkpointId });

        response.Status.Should().Be(HttpStatusCode.UnprocessableEntity);
        ErrorCode(response).Should().Be("FUNCTION_INPUT_INVALID");
    }

    [Fact]
    public async Task UnknownCheckpoint_IsReportedDistinctly()
    {
        var response = await InvokeAsync("evaluate-network", new { checkpoint_id = "no-such-checkpoint" });

        response.Status.Should().Be(HttpStatusCode.UnprocessableEntity);
        ErrorCode(response).Should().Be("CHECKPOINT_NOT_FOUND");
    }

    [Fact]
    public async Task HealthEndpoint_ReportsTheServedContract()
    {
        using var client = _factory.CreateClient();

        var payload = JsonDocument.Parse(await client.GetStringAsync(new Uri("/health", UriKind.Relative)));

        payload.RootElement.GetProperty("status").GetString().Should().Be("ok");
        payload.RootElement.GetProperty("function_namespace").GetString()
            .Should().Be(FunctionHostFactory.FunctionNamespace);
        payload.RootElement.GetProperty("operations").EnumerateArray()
            .Select(operation => operation.GetString())
            .Should().Contain("train-epoch");
    }

    private static string NewRunId() => $"test{Guid.NewGuid():N}";

    private static string Now() =>
        DateTimeOffset.UtcNow.ToUnixTimeMilliseconds().ToString(System.Globalization.CultureInfo.InvariantCulture);

    private async Task<InvocationResponse> InvokeAsync(string operation, object input)
    {
        var invocationId = $"inv-{Guid.NewGuid():N}";
        var body = FunctionHostFactory.BodyFor(operation, input, invocationId);
        var timestamp = Now();

        var request = Request(operation, body, operation);
        request.Headers.Add("x-ductape-invocation-id", invocationId);
        request.Headers.Add("x-ductape-timestamp", timestamp);
        request.Headers.Add("x-ductape-signature", FunctionHostFactory.Sign(timestamp, body));

        return await SendAsync(request);
    }

    private static HttpRequestMessage Request(string routeOperation, string body, string headerOperation)
    {
        var request = new HttpRequestMessage(
            HttpMethod.Post,
            new Uri(FunctionHostFactory.RouteFor(routeOperation), UriKind.Relative))
        {
            Content = new StringContent(body, Encoding.UTF8),
        };

        request.Content.Headers.ContentType = new MediaTypeHeaderValue("application/json");
        request.Headers.Add(
            "x-ductape-function",
            $"{FunctionHostFactory.FunctionNamespace}.{headerOperation}@{FunctionHostFactory.FunctionVersion}");

        return request;
    }

    private async Task<InvocationResponse> SendAsync(HttpRequestMessage request)
    {
        using var client = _factory.CreateClient();
        using var response = await client.SendAsync(request);

        return new InvocationResponse(response.StatusCode, await response.Content.ReadAsStringAsync());
    }

    private static JsonElement Output(InvocationResponse response) =>
        JsonDocument.Parse(response.Body).RootElement.GetProperty("output");

    private static string? ErrorCode(InvocationResponse response) =>
        JsonDocument.Parse(response.Body).RootElement.GetProperty("error").GetProperty("code").GetString();

    private static string? ErrorMessage(InvocationResponse response) =>
        JsonDocument.Parse(response.Body).RootElement.GetProperty("error").GetProperty("message").GetString();

    private sealed record InvocationResponse(HttpStatusCode Status, string Body);
}
