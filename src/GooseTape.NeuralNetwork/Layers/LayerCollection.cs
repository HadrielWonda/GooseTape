using System.Collections;
using GooseTape.NeuralNetwork.Maths;

namespace GooseTape.NeuralNetwork.Layers;

/// <summary>
/// The ordered dense layers of a network, from input side to output side.
/// </summary>
public sealed class LayerCollection : IReadOnlyList<DenseLayer>
{
    private readonly IReadOnlyList<DenseLayer> _layers;

    private LayerCollection(IReadOnlyList<DenseLayer> layers) => _layers = layers;

    /// <summary>Gets the number of layers.</summary>
    public int Count => _layers.Count;

    /// <summary>Gets the layer at <paramref name="index"/>.</summary>
    /// <param name="index">The zero based layer index.</param>
    /// <returns>The layer at that position.</returns>
    public DenseLayer this[int index] => _layers[index];

    /// <summary>
    /// Creates a validated layer collection.
    /// </summary>
    /// <param name="layers">The layers, ordered from input side to output side.</param>
    /// <returns>A new <see cref="LayerCollection"/>.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="layers"/> is null.</exception>
    /// <exception cref="ArgumentException">Thrown when fewer than two layers are supplied.</exception>
    public static LayerCollection From(IReadOnlyList<DenseLayer> layers)
    {
        ArgumentNullException.ThrowIfNull(layers);

        if (layers.Count < 2)
        {
            throw new ArgumentException(
                $"A network requires at least one hidden layer and one output layer. Received {layers.Count}.",
                nameof(layers));
        }

        return new LayerCollection([.. layers]);
    }

    /// <summary>
    /// Runs a batch through every layer in order, recording each pass.
    /// </summary>
    /// <param name="input">The batch input, of shape <c>batch x inputs</c>.</param>
    /// <returns>The trace of every layer's forward pass.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="input"/> is null.</exception>
    public ForwardTrace Forward(Matrix input)
    {
        ArgumentNullException.ThrowIfNull(input);

        var passes = new List<LayerForwardPass>(_layers.Count);
        var current = input;

        foreach (var layer in _layers)
        {
            var pass = layer.Forward(current);
            passes.Add(pass);
            current = pass.Activation;
        }

        return ForwardTrace.From(passes);
    }

    /// <summary>
    /// Returns a copy of this collection with each layer's gradient applied.
    /// </summary>
    /// <param name="gradients">One gradient per layer, in the same order.</param>
    /// <param name="stepSize">The learning rate to scale each gradient by.</param>
    /// <returns>A new collection holding the updated layers.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="gradients"/> is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the gradient count does not match the layer count.</exception>
    public LayerCollection Descend(IReadOnlyList<LayerGradient> gradients, double stepSize)
    {
        ArgumentNullException.ThrowIfNull(gradients);

        if (gradients.Count != _layers.Count)
        {
            throw new ArgumentException(
                $"Expected {_layers.Count} gradients to match the layer count. Received {gradients.Count}.",
                nameof(gradients));
        }

        var descended = new DenseLayer[_layers.Count];

        for (var index = 0; index < _layers.Count; index++)
        {
            descended[index] = _layers[index].Descend(gradients[index], stepSize);
        }

        return new LayerCollection(descended);
    }

    /// <inheritdoc />
    public IEnumerator<DenseLayer> GetEnumerator() => _layers.GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}
