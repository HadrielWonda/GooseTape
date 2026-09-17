namespace GooseTape.Functions.Host.Ductape;

/// <summary>
/// The body returned when an invocation succeeds.
/// </summary>
/// <param name="InvocationId">The identifier echoed from the request context.</param>
/// <param name="Output">The operation output.</param>
public sealed record PortableFunctionSuccess(string InvocationId, object Output);

/// <summary>
/// The body returned when an invocation fails.
/// </summary>
/// <param name="InvocationId">The identifier echoed from the request, or empty when unknown.</param>
/// <param name="Error">What went wrong.</param>
public sealed record PortableFunctionRejection(string InvocationId, PortableFunctionFailure Error);
