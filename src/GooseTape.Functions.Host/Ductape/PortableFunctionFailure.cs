namespace GooseTape.Functions.Host.Ductape;

/// <summary>
/// The error shape Ductape expects when an invocation cannot be completed.
/// </summary>
/// <param name="Code">A stable machine readable failure code.</param>
/// <param name="Message">A human readable description, free of secrets and personal data.</param>
/// <param name="Retryable">Whether Ductape may retry the invocation as-is.</param>
public sealed record PortableFunctionFailure(string Code, string Message, bool Retryable);
