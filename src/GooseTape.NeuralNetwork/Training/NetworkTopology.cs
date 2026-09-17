using System.Collections;

namespace GooseTape.NeuralNetwork.Training;

/// <summary>
/// The neuron counts of a network, from the input layer through to the output layer.
/// </summary>
/// <remarks>
/// For MNIST the canonical topology is <c>784, 128, 10</c>: one input per pixel, one hidden
/// layer, and one output per digit.
/// </remarks>
public sealed class NetworkTopology : IReadOnlyList<LayerSize>
{
    private readonly IReadOnlyList<LayerSize> _sizes;

    private NetworkTopology(IReadOnlyList<LayerSize> sizes) => _sizes = sizes;

    /// <summary>Gets the number of layer sizes, which is one more than the number of dense layers.</summary>
    public int Count => _sizes.Count;

    /// <summary>Gets the layer size at <paramref name="index"/>.</summary>
    /// <param name="index">The zero based position.</param>
    /// <returns>The size at that position.</returns>
    public LayerSize this[int index] => _sizes[index];

    /// <summary>Gets the number of inputs the network accepts.</summary>
    public LayerSize InputSize => _sizes[0];

    /// <summary>Gets the number of classes the network predicts.</summary>
    public LayerSize OutputSize => _sizes[^1];

    /// <summary>Gets the number of dense layers this topology describes.</summary>
    public int DenseLayerCount => _sizes.Count - 1;

    /// <summary>
    /// Creates a validated topology from raw neuron counts.
    /// </summary>
    /// <param name="sizes">
    /// The neuron counts, from input to output. At least three are required, so that the network
    /// has an input size, at least one hidden layer, and an output size.
    /// </param>
    /// <returns>A new <see cref="NetworkTopology"/>.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="sizes"/> is null.</exception>
    /// <exception cref="ArgumentException">Thrown when fewer than three sizes are supplied.</exception>
    public static NetworkTopology Create(IReadOnlyList<int> sizes)
    {
        ArgumentNullException.ThrowIfNull(sizes);

        if (sizes.Count < 3)
        {
            throw new ArgumentException(
                $"A topology needs an input size, at least one hidden size, and an output size. Received {sizes.Count}.",
                nameof(sizes));
        }

        return new NetworkTopology([.. sizes.Select(LayerSize.Create)]);
    }

    /// <inheritdoc />
    public IEnumerator<LayerSize> GetEnumerator() => _sizes.GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}
