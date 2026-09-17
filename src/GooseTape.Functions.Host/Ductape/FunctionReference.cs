namespace GooseTape.Functions.Host.Ductape;

/// <summary>
/// Identifies which portable function operation an invocation is addressed to.
/// </summary>
/// <param name="Namespace">The contract namespace, such as <c>goosetape.neural-network</c>.</param>
/// <param name="Operation">The operation within the contract, such as <c>train-epoch</c>.</param>
/// <param name="Version">The contract version.</param>
public sealed record FunctionReference(string Namespace, string Operation, string Version)
{
    /// <summary>
    /// Renders the reference the way the <c>x-ductape-function</c> header spells it.
    /// </summary>
    /// <returns>The reference as <c>namespace.operation@version</c>.</returns>
    public string ToHeaderValue() => $"{Namespace}.{Operation}@{Version}";
}
