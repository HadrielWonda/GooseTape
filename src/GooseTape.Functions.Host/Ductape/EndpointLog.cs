namespace GooseTape.Functions.Host.Ductape;

/// <summary>
/// Source generated log messages for the portable function endpoint.
/// </summary>
/// <remarks>
/// Compile-time generated delegates keep the message templates and their named fields fixed, so
/// every record reaches the log with the same structure and nothing is boxed or formatted when
/// the level is disabled.
/// </remarks>
internal static partial class EndpointLog
{
    [LoggerMessage(
        EventId = 1000,
        Level = LogLevel.Warning,
        Message = "Rejected portable function invocation {InvocationId} for {Operation}: signature invalid or expired.")]
    public static partial void SignatureRejected(ILogger logger, string invocationId, string operation);

    [LoggerMessage(
        EventId = 1001,
        Level = LogLevel.Warning,
        Message = "Rejected portable function invocation {InvocationId}: {Reason}")]
    public static partial void InvocationRejected(ILogger logger, string invocationId, string reason);

    [LoggerMessage(
        EventId = 1002,
        Level = LogLevel.Information,
        Message = "Completed {Operation} in {ElapsedMilliseconds}ms.")]
    public static partial void OperationCompleted(ILogger logger, string operation, double elapsedMilliseconds);

    [LoggerMessage(
        EventId = 1003,
        Level = LogLevel.Warning,
        Message = "Operation {Operation} failed with {Code} after {ElapsedMilliseconds}ms.")]
    public static partial void OperationFailed(
        ILogger logger,
        Exception exception,
        string operation,
        string code,
        double elapsedMilliseconds);
}
