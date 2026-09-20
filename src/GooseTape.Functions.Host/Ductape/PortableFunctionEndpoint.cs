using System.Text.Json;
using Microsoft.AspNetCore.Http.Features;

namespace GooseTape.Functions.Host.Ductape;

/// <summary>
/// The signed HTTP endpoint Ductape invokes portable functions through.
/// </summary>
/// <remarks>
/// The verification order matters. The signature is checked against the raw body before the
/// body is parsed, so malformed or hostile payloads are rejected before any deserialisation
/// happens. Route segments and the function header are then checked against the signed body, so
/// a validly signed request cannot be redirected at a different operation in flight.
/// </remarks>
public static class PortableFunctionEndpoint
{
    /// <summary>The route Ductape derives from the configured function base URL.</summary>
    public const string RoutePattern = "/.well-known/ductape/functions/{contractNamespace}/{version}/{operation}";

    /// <summary>
    /// The largest invocation body this endpoint accepts.
    /// </summary>
    /// <remarks>
    /// Weights never cross this boundary, so a legitimate payload is small: the widest is a
    /// classify request, at roughly 800 KB for the maximum supported image. Kestrel's default of
    /// 30 MB is far more than that, and this endpoint is reachable by anything that can open a
    /// socket, so the limit is tightened here and applied before the body is read.
    /// </remarks>
    public const long MaxRequestBodyBytes = 2L * 1024 * 1024;

    private const string TimestampHeader = "x-ductape-timestamp";
    private const string SignatureHeader = "x-ductape-signature";
    private const string InvocationHeader = "x-ductape-invocation-id";
    private const string FunctionHeader = "x-ductape-function";

    /// <summary>
    /// Handles one portable function invocation.
    /// </summary>
    /// <param name="httpContext">The request being served.</param>
    /// <param name="contractNamespace">The namespace segment of the route.</param>
    /// <param name="version">The version segment of the route.</param>
    /// <param name="operation">The operation segment of the route.</param>
    /// <param name="catalogue">The operations this host serves.</param>
    /// <param name="verifier">Verifies the request signature.</param>
    /// <param name="ledger">Keeps each invocation identifier single-use within the replay window.</param>
    /// <param name="clock">Supplies the current time for the replay window.</param>
    /// <param name="loggerFactory">Creates the logger used for correlated logging.</param>
    /// <param name="cancellationToken">Token to monitor for cancellation requests.</param>
    /// <returns>The invocation response, in the shape the Ductape SDK expects.</returns>
    /// <exception cref="ArgumentNullException">Thrown when any injected dependency is null.</exception>
    public static async Task<IResult> HandleAsync(
        HttpContext httpContext,
        string contractNamespace,
        string version,
        string operation,
        PortableFunctionCatalogue catalogue,
        InvocationSignatureVerifier verifier,
        InvocationLedger ledger,
        TimeProvider clock,
        ILoggerFactory loggerFactory,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(httpContext);
        ArgumentNullException.ThrowIfNull(catalogue);
        ArgumentNullException.ThrowIfNull(verifier);
        ArgumentNullException.ThrowIfNull(ledger);
        ArgumentNullException.ThrowIfNull(clock);
        ArgumentNullException.ThrowIfNull(loggerFactory);

        var logger = loggerFactory.CreateLogger(LoggerCategory);
        var suppliedInvocationId = HeaderOrEmpty(httpContext, InvocationHeader);

        if (IsTooLarge(httpContext))
        {
            EndpointLog.BodyTooLarge(logger, suppliedInvocationId, operation, httpContext.Request.ContentLength ?? -1);

            return Rejection(
                StatusCodes.Status413PayloadTooLarge,
                suppliedInvocationId,
                new PortableFunctionFailure(
                    "FUNCTION_REQUEST_TOO_LARGE",
                    $"Function request body exceeds the {MaxRequestBodyBytes} byte limit.",
                    Retryable: false));
        }

        var body = await ReadBodyAsync(httpContext).ConfigureAwait(false);

        var authentic = verifier.IsValid(
            HeaderOrNull(httpContext, TimestampHeader),
            HeaderOrNull(httpContext, SignatureHeader),
            body,
            clock.GetUtcNow());

        if (!authentic)
        {
            EndpointLog.SignatureRejected(logger, suppliedInvocationId, operation);

            return Rejection(
                StatusCodes.Status401Unauthorized,
                suppliedInvocationId,
                new PortableFunctionFailure(
                    "FUNCTION_SIGNATURE_INVALID",
                    "Function invocation signature is invalid or expired.",
                    Retryable: false));
        }

        return await DispatchAsync(
                new RequestEnvelope(body, contractNamespace, version, operation, suppliedInvocationId),
                httpContext,
                catalogue,
                ledger,
                clock,
                logger,
                cancellationToken)
            .ConfigureAwait(false);
    }

    private const string LoggerCategory = "GooseTape.Functions.Host.Ductape.PortableFunctionEndpoint";

    private static async Task<IResult> DispatchAsync(
        RequestEnvelope envelope,
        HttpContext httpContext,
        PortableFunctionCatalogue catalogue,
        InvocationLedger ledger,
        TimeProvider clock,
        ILogger logger,
        CancellationToken cancellationToken)
    {
        var invocation = Parse(envelope.Body);

        if (invocation is null)
        {
            return Rejection(
                StatusCodes.Status400BadRequest,
                envelope.SuppliedInvocationId,
                new PortableFunctionFailure(
                    "FUNCTION_REQUEST_INVALID",
                    "Function request body is not valid JSON, or is missing its function reference.",
                    Retryable: false));
        }

        var mismatch = RouteMismatch(invocation, envelope, httpContext);

        if (mismatch is not null)
        {
            EndpointLog.InvocationRejected(logger, invocation.Context.InvocationId, mismatch);

            return Rejection(
                StatusCodes.Status422UnprocessableEntity,
                invocation.Context.InvocationId,
                new PortableFunctionFailure("FUNCTION_ROUTE_MISMATCH", mismatch, Retryable: false));
        }

        var recorded = await ExecuteThroughLedgerAsync(invocation, catalogue, ledger, clock, logger, cancellationToken)
            .ConfigureAwait(false);

        return Results.Json(recorded.Payload, DuctapeJson.Options, statusCode: recorded.StatusCode);
    }

    /// <summary>
    /// Runs the invocation once per identifier, replaying the recorded response for repeats.
    /// </summary>
    /// <remarks>
    /// An invocation with no identifier cannot be deduplicated, so it simply executes. Ductape
    /// always supplies one.
    /// </remarks>
    private static Task<RecordedInvocation> ExecuteThroughLedgerAsync(
        PortableFunctionInvocation invocation,
        PortableFunctionCatalogue catalogue,
        InvocationLedger ledger,
        TimeProvider clock,
        ILogger logger,
        CancellationToken cancellationToken)
    {
        var invocationId = invocation.Context.InvocationId;

        if (string.IsNullOrWhiteSpace(invocationId))
        {
            return ExecuteAsync(invocation, catalogue, clock, logger, cancellationToken);
        }

        if (ledger.Contains(invocationId))
        {
            EndpointLog.InvocationReplayed(logger, invocationId, invocation.Function.Operation);
        }

        return ledger.ExecuteOnceAsync(
            invocationId,
            () => ExecuteAsync(invocation, catalogue, clock, logger, cancellationToken));
    }

    private static async Task<RecordedInvocation> ExecuteAsync(
        PortableFunctionInvocation invocation,
        PortableFunctionCatalogue catalogue,
        TimeProvider clock,
        ILogger logger,
        CancellationToken cancellationToken)
    {
        using var scope = logger.BeginScope(ScopeFor(invocation));
        var startedAt = clock.GetTimestamp();

        try
        {
            var target = catalogue.Resolve(invocation.Function);

            var output = await target
                .ExecuteAsync(invocation.Input, invocation.Context, cancellationToken)
                .ConfigureAwait(false);

            var elapsedMilliseconds = clock.GetElapsedTime(startedAt).TotalMilliseconds;
            EndpointLog.OperationCompleted(logger, invocation.Function.Operation, elapsedMilliseconds);

            return new RecordedInvocation(
                StatusCodes.Status200OK,
                new PortableFunctionSuccess(invocation.Context.InvocationId, output));
        }
        catch (PortableFunctionException failure)
        {
            var elapsedMilliseconds = clock.GetElapsedTime(startedAt).TotalMilliseconds;
            EndpointLog.OperationFailed(logger, failure, invocation.Function.Operation, failure.Code, elapsedMilliseconds);

            return new RecordedInvocation(
                StatusFor(failure.Code),
                new PortableFunctionRejection(
                    invocation.Context.InvocationId,
                    new PortableFunctionFailure(failure.Code, failure.Message, failure.Retryable)));
        }
    }

    private static Dictionary<string, object> ScopeFor(PortableFunctionInvocation invocation) => new(StringComparer.Ordinal)
    {
        ["InvocationId"] = invocation.Context.InvocationId,
        ["Operation"] = invocation.Function.Operation,
        ["FeatureTag"] = invocation.Context.FeatureTag ?? "unknown",
        ["FeatureRunId"] = invocation.Context.FeatureRunId ?? "unknown",
        ["StepTag"] = invocation.Context.StepTag ?? "unknown",
        ["Env"] = invocation.Context.Env ?? "unknown",
    };

    private static int StatusFor(string code) => code switch
    {
        "FUNCTION_UNAVAILABLE" => StatusCodes.Status503ServiceUnavailable,
        _ => StatusCodes.Status422UnprocessableEntity,
    };

    private static string? RouteMismatch(
        PortableFunctionInvocation invocation,
        RequestEnvelope envelope,
        HttpContext httpContext)
    {
        var reference = invocation.Function;

        var addressesThisRoute =
            string.Equals(reference.Namespace, envelope.ContractNamespace, StringComparison.Ordinal)
            && string.Equals(reference.Version, envelope.Version, StringComparison.Ordinal)
            && string.Equals(reference.Operation, envelope.Operation, StringComparison.Ordinal);

        if (!addressesThisRoute)
        {
            return "Route parameters do not match the signed invocation body.";
        }

        var declared = HeaderOrNull(httpContext, FunctionHeader);

        return string.Equals(declared, reference.ToHeaderValue(), StringComparison.Ordinal)
            ? null
            : "The x-ductape-function header does not match the signed invocation body.";
    }

    private static PortableFunctionInvocation? Parse(string body)
    {
        try
        {
            var invocation = JsonSerializer.Deserialize<PortableFunctionInvocation>(body, DuctapeJson.Options);

            return invocation?.Function is null || invocation.Context is null ? null : invocation;
        }
        catch (JsonException)
        {
            return null;
        }
    }

    /// <summary>
    /// Rejects an over-large body before it is read, and caps what a chunked request may stream.
    /// </summary>
    /// <remarks>
    /// A declared Content-Length is checked directly. A chunked request declares no length, so
    /// the per-request Kestrel limit is lowered instead and enforced as the body is read.
    /// </remarks>
    private static bool IsTooLarge(HttpContext httpContext)
    {
        var sizeFeature = httpContext.Features.Get<IHttpMaxRequestBodySizeFeature>();

        if (sizeFeature is { IsReadOnly: false })
        {
            sizeFeature.MaxRequestBodySize = MaxRequestBodyBytes;
        }

        return httpContext.Request.ContentLength > MaxRequestBodyBytes;
    }

    private static async Task<string> ReadBodyAsync(HttpContext httpContext)
    {
        using var reader = new StreamReader(httpContext.Request.Body, leaveOpen: true);

        return await reader.ReadToEndAsync().ConfigureAwait(false);
    }

    private static IResult Rejection(int statusCode, string invocationId, PortableFunctionFailure failure) =>
        Results.Json(new PortableFunctionRejection(invocationId, failure), DuctapeJson.Options, statusCode: statusCode);

    private static string? HeaderOrNull(HttpContext httpContext, string name) =>
        httpContext.Request.Headers.TryGetValue(name, out var values) ? values.FirstOrDefault() : null;

    private static string HeaderOrEmpty(HttpContext httpContext, string name) =>
        HeaderOrNull(httpContext, name) ?? string.Empty;

    private sealed record RequestEnvelope(
        string Body,
        string ContractNamespace,
        string Version,
        string Operation,
        string SuppliedInvocationId);
}
