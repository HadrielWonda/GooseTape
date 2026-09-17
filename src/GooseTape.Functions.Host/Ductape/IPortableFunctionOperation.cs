using System.Text.Json;

namespace GooseTape.Functions.Host.Ductape;

/// <summary>
/// One operation exposed to Ductape as part of a portable function contract.
/// </summary>
public interface IPortableFunctionOperation
{
    /// <summary>Gets the operation name, matching the contract declared on the Ductape side.</summary>
    string Name { get; }

    /// <summary>
    /// Runs the operation.
    /// </summary>
    /// <param name="input">The raw operation input, as sent by the calling feature step.</param>
    /// <param name="context">The execution context of the calling feature step.</param>
    /// <param name="cancellationToken">Token to monitor for cancellation requests.</param>
    /// <returns>The operation output, serialised into the invocation response.</returns>
    /// <exception cref="PortableFunctionException">Thrown when the operation cannot complete.</exception>
    Task<object> ExecuteAsync(JsonElement input, InvocationContext context, CancellationToken cancellationToken);
}
