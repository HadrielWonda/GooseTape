namespace GooseTape.Functions.Host.Ductape;

/// <summary>
/// The response produced for one invocation, retained so a replay of the same invocation
/// identifier returns exactly what the first attempt returned.
/// </summary>
/// <param name="StatusCode">The HTTP status the invocation produced.</param>
/// <param name="Payload">The response body, serialised the same way on every replay.</param>
public sealed record RecordedInvocation(int StatusCode, object Payload);
