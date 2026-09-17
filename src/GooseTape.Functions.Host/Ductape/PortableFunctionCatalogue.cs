namespace GooseTape.Functions.Host.Ductape;

/// <summary>
/// The portable function contract this host serves: one namespace, one version, and the
/// operations within it.
/// </summary>
public sealed class PortableFunctionCatalogue
{
    private readonly FunctionContract _contract;
    private readonly IReadOnlyDictionary<string, IPortableFunctionOperation> _operations;

    private PortableFunctionCatalogue(
        FunctionContract contract,
        IReadOnlyDictionary<string, IPortableFunctionOperation> operations)
    {
        _contract = contract;
        _operations = operations;
    }

    /// <summary>Gets the namespace this host serves.</summary>
    public string Namespace => _contract.Namespace;

    /// <summary>Gets the contract version this host serves.</summary>
    public string Version => _contract.Version;

    /// <summary>Gets the names of every operation served, for diagnostics.</summary>
    public IEnumerable<string> OperationNames => _operations.Keys;

    /// <summary>
    /// Creates a catalogue.
    /// </summary>
    /// <param name="contractNamespace">The contract namespace.</param>
    /// <param name="version">The contract version.</param>
    /// <param name="operations">The operations served, keyed by their own names.</param>
    /// <returns>A new <see cref="PortableFunctionCatalogue"/>.</returns>
    /// <exception cref="ArgumentException">Thrown when the namespace or version is blank, or no operations are supplied.</exception>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="operations"/> is null.</exception>
    public static PortableFunctionCatalogue Create(
        string contractNamespace,
        string version,
        IReadOnlyCollection<IPortableFunctionOperation> operations)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(contractNamespace);
        ArgumentException.ThrowIfNullOrWhiteSpace(version);
        ArgumentNullException.ThrowIfNull(operations);

        if (operations.Count == 0)
        {
            throw new ArgumentException("A catalogue must serve at least one operation.", nameof(operations));
        }

        return new PortableFunctionCatalogue(
            new FunctionContract(contractNamespace, version),
            operations.ToDictionary(operation => operation.Name, StringComparer.Ordinal));
    }

    /// <summary>
    /// Resolves the operation an invocation addresses.
    /// </summary>
    /// <param name="reference">The function reference taken from the signed request body.</param>
    /// <returns>The matching operation.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="reference"/> is null.</exception>
    /// <exception cref="PortableFunctionException">
    /// Thrown when the reference names a different contract, or an operation this host does not serve.
    /// </exception>
    public IPortableFunctionOperation Resolve(FunctionReference reference)
    {
        ArgumentNullException.ThrowIfNull(reference);

        if (!string.Equals(reference.Namespace, _contract.Namespace, StringComparison.Ordinal)
            || !string.Equals(reference.Version, _contract.Version, StringComparison.Ordinal))
        {
            throw new PortableFunctionException(
                "FUNCTION_UNAVAILABLE",
                $"This host serves {_contract.Namespace}@{_contract.Version}, not {reference.Namespace}@{reference.Version}.",
                retryable: false);
        }

        if (!_operations.TryGetValue(reference.Operation, out var operation))
        {
            throw new PortableFunctionException(
                "FUNCTION_UNAVAILABLE",
                $"No operation named {reference.Operation} is served. Known operations: {string.Join(", ", _operations.Keys)}.",
                retryable: false);
        }

        return operation;
    }

    private sealed record FunctionContract(string Namespace, string Version);
}
