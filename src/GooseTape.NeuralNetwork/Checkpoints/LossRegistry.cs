using GooseTape.NeuralNetwork.Losses;

namespace GooseTape.NeuralNetwork.Checkpoints;

/// <summary>
/// Resolves loss identifiers read from a checkpoint back into loss function instances.
/// </summary>
public sealed class LossRegistry
{
    private readonly IReadOnlyDictionary<string, Func<ILossFunction>> _factories;

    private LossRegistry(IReadOnlyDictionary<string, Func<ILossFunction>> factories) => _factories = factories;

    /// <summary>Gets a registry holding every loss function this library ships with.</summary>
    public static LossRegistry Default { get; } = new(
        new Dictionary<string, Func<ILossFunction>>(StringComparer.Ordinal)
        {
            [CrossEntropyLoss.Identifier] = static () => new CrossEntropyLoss(),
        });

    /// <summary>
    /// Creates a registry from the supplied factories.
    /// </summary>
    /// <param name="factories">Loss identifier to factory, keyed ordinally.</param>
    /// <returns>A new <see cref="LossRegistry"/>.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="factories"/> is null.</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="factories"/> is empty.</exception>
    public static LossRegistry From(IReadOnlyDictionary<string, Func<ILossFunction>> factories)
    {
        ArgumentNullException.ThrowIfNull(factories);

        if (factories.Count == 0)
        {
            throw new ArgumentException("A loss registry requires at least one factory.", nameof(factories));
        }

        return new LossRegistry(factories);
    }

    /// <summary>
    /// Resolves a loss function by identifier.
    /// </summary>
    /// <param name="name">The identifier persisted in the checkpoint.</param>
    /// <returns>A new loss function instance.</returns>
    /// <exception cref="ArgumentException">Thrown when <paramref name="name"/> is null or blank.</exception>
    /// <exception cref="NotSupportedException">Thrown when no loss is registered under that identifier.</exception>
    public ILossFunction Resolve(string name)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);

        if (!_factories.TryGetValue(name, out var factory))
        {
            throw new NotSupportedException(
                $"No loss function is registered under the identifier {name}. Known identifiers: {string.Join(", ", _factories.Keys)}.");
        }

        return factory();
    }
}
