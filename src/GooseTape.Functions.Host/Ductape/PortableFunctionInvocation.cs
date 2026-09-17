using System.Text.Json;

namespace GooseTape.Functions.Host.Ductape;

/// <summary>
/// The body Ductape posts when invoking a portable function over signed HTTP.
/// </summary>
/// <param name="Function">Which operation is being invoked.</param>
/// <param name="Input">
/// The operation input, left as raw JSON so each operation can bind it to its own contract.
/// </param>
/// <param name="Context">The execution context of the calling feature step.</param>
public sealed record PortableFunctionInvocation(
    FunctionReference Function,
    JsonElement Input,
    InvocationContext Context);
