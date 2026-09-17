using GooseTape.NeuralNetwork.Activations;

namespace GooseTape.NeuralNetwork.Checkpoints;

/// <summary>
/// Resolves activation identifiers read from a checkpoint back into activation instances.
/// </summary>
/// <remarks>
/// New activations are registered here rather than by extending a switch inside the
/// serialiser, so adding one never means reopening the serialisation code.
/// </remarks>
public sealed class ActivationRegistry
{
    private readonly IReadOnlyDictionary<string, Func<IActivationFunction>> _factories;

    private ActivationRegistry(IReadOnlyDictionary<string, Func<IActivationFunction>> factories) =>
        _factories = factories;

    /// <summary>Gets a registry holding every activation this library ships with.</summary>
    public static ActivationRegistry Default { get; } = new(
        new Dictionary<string, Func<IActivationFunction>>(StringComparer.Ordinal)
        {
            [RectifiedLinearActivation.Identifier] = static () => new RectifiedLinearActivation(),
            [SoftmaxActivation.Identifier] = static () => new SoftmaxActivation(),
        });

    /// <summary>
    /// Creates a registry from the supplied factories.
    /// </summary>
    /// <param name="factories">Activation identifier to factory, keyed ordinally.</param>
    /// <returns>A new <see cref="ActivationRegistry"/>.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="factories"/> is null.</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="factories"/> is empty.</exception>
    public static ActivationRegistry From(IReadOnlyDictionary<string, Func<IActivationFunction>> factories)
    {
        ArgumentNullException.ThrowIfNull(factories);

        if (factories.Count == 0)
        {
            throw new ArgumentException("An activation registry requires at least one factory.", nameof(factories));
        }

        return new ActivationRegistry(factories);
    }

    /// <summary>
    /// Resolves an activation by identifier.
    /// </summary>
    /// <param name="name">The identifier persisted in the checkpoint.</param>
    /// <returns>A new activation instance.</returns>
    /// <exception cref="ArgumentException">Thrown when <paramref name="name"/> is null or blank.</exception>
    /// <exception cref="NotSupportedException">Thrown when no activation is registered under that identifier.</exception>
    public IActivationFunction Resolve(string name)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);

        if (!_factories.TryGetValue(name, out var factory))
        {
            throw new NotSupportedException(
                $"No activation is registered under the identifier {name}. Known identifiers: {string.Join(", ", _factories.Keys)}.");
        }

        return factory();
    }
}
