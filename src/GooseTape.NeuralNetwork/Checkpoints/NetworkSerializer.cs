using GooseTape.NeuralNetwork.Layers;
using GooseTape.NeuralNetwork.Maths;
using GooseTape.NeuralNetwork.Training;

namespace GooseTape.NeuralNetwork.Checkpoints;

/// <summary>
/// Converts a network to and from the flattened form a checkpoint stores.
/// </summary>
public sealed class NetworkSerializer
{
    private readonly ActivationRegistry _activations;
    private readonly LossRegistry _losses;

    private NetworkSerializer(ActivationRegistry activations, LossRegistry losses)
    {
        _activations = activations;
        _losses = losses;
    }

    /// <summary>Gets a serialiser bound to the registries this library ships with.</summary>
    public static NetworkSerializer Default { get; } = new(ActivationRegistry.Default, LossRegistry.Default);

    /// <summary>
    /// Creates a serialiser over the supplied registries.
    /// </summary>
    /// <param name="activations">The activation registry.</param>
    /// <param name="losses">The loss registry.</param>
    /// <returns>A new <see cref="NetworkSerializer"/>.</returns>
    /// <exception cref="ArgumentNullException">Thrown when either registry is null.</exception>
    public static NetworkSerializer From(ActivationRegistry activations, LossRegistry losses)
    {
        ArgumentNullException.ThrowIfNull(activations);
        ArgumentNullException.ThrowIfNull(losses);

        return new NetworkSerializer(activations, losses);
    }

    /// <summary>
    /// Flattens a network into a storable snapshot.
    /// </summary>
    /// <param name="network">The network to flatten.</param>
    /// <returns>The snapshot.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="network"/> is null.</exception>
    public static NetworkSnapshot ToSnapshot(FeedForwardNetwork network)
    {
        ArgumentNullException.ThrowIfNull(network);

        return new NetworkSnapshot([.. network.Layers.Select(ToLayerSnapshot)], network.Loss.Name);
    }

    /// <summary>
    /// Rebuilds a network from a snapshot.
    /// </summary>
    /// <param name="snapshot">The snapshot to rebuild from.</param>
    /// <returns>The rebuilt network.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="snapshot"/> is null.</exception>
    /// <exception cref="InvalidDataException">
    /// Thrown when the snapshot holds no layers, or a layer weight or bias count disagrees with
    /// its declared shape.
    /// </exception>
    /// <exception cref="NotSupportedException">
    /// Thrown when the snapshot names an activation or loss this serialiser cannot resolve.
    /// </exception>
    public FeedForwardNetwork FromSnapshot(NetworkSnapshot snapshot)
    {
        ArgumentNullException.ThrowIfNull(snapshot);

        if (snapshot.Layers.Count == 0)
        {
            throw new InvalidDataException("A network snapshot must describe at least one layer.");
        }

        return FeedForwardNetwork.Create(
            LayerCollection.From([.. snapshot.Layers.Select(ToDenseLayer)]),
            _losses.Resolve(snapshot.LossFunction));
    }

    private static LayerSnapshot ToLayerSnapshot(DenseLayer layer) => new(
        layer.Parameters.Weights.Shape.RowCount,
        layer.Parameters.Weights.Shape.ColumnCount,
        layer.Activation.Name,
        layer.Parameters.Weights.ToArray(),
        layer.Parameters.Biases.ToArray());

    private DenseLayer ToDenseLayer(LayerSnapshot snapshot)
    {
        GuardLayerCounts(snapshot);

        var parameters = LayerParameters.Create(
            Matrix.FromValues(MatrixShape.Create(snapshot.InputCount, snapshot.NeuronCount), [.. snapshot.Weights]),
            Matrix.FromValues(MatrixShape.Create(1, snapshot.NeuronCount), [.. snapshot.Biases]));

        return DenseLayer.Create(parameters, _activations.Resolve(snapshot.Activation));
    }

    private static void GuardLayerCounts(LayerSnapshot snapshot)
    {
        var expectedWeightCount = snapshot.InputCount * snapshot.NeuronCount;

        if (snapshot.Weights.Count != expectedWeightCount)
        {
            throw new InvalidDataException(
                $"A {snapshot.InputCount}x{snapshot.NeuronCount} layer needs {expectedWeightCount} weights but the snapshot holds {snapshot.Weights.Count}.");
        }

        if (snapshot.Biases.Count != snapshot.NeuronCount)
        {
            throw new InvalidDataException(
                $"A layer of {snapshot.NeuronCount} neurons needs {snapshot.NeuronCount} biases but the snapshot holds {snapshot.Biases.Count}.");
        }
    }
}
