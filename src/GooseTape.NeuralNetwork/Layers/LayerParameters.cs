using GooseTape.NeuralNetwork.Maths;

namespace GooseTape.NeuralNetwork.Layers;

/// <summary>
/// The learnable state of a dense layer: its weight matrix and its bias row.
/// </summary>
public sealed class LayerParameters
{
    private readonly Matrix _weights;
    private readonly Matrix _biases;

    private LayerParameters(Matrix weights, Matrix biases)
    {
        _weights = weights;
        _biases = biases;
    }

    /// <summary>Gets the weight matrix, of shape <c>inputs x neurons</c>.</summary>
    public Matrix Weights => _weights;

    /// <summary>Gets the bias row, of shape <c>1 x neurons</c>.</summary>
    public Matrix Biases => _biases;

    /// <summary>
    /// Creates a validated parameter set.
    /// </summary>
    /// <param name="weights">The weight matrix, of shape <c>inputs x neurons</c>.</param>
    /// <param name="biases">The bias row, of shape <c>1 x neurons</c>.</param>
    /// <returns>A new <see cref="LayerParameters"/>.</returns>
    /// <exception cref="ArgumentNullException">Thrown when either argument is null.</exception>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="biases"/> is not a single row matching the width of
    /// <paramref name="weights"/>.
    /// </exception>
    public static LayerParameters Create(Matrix weights, Matrix biases)
    {
        ArgumentNullException.ThrowIfNull(weights);
        ArgumentNullException.ThrowIfNull(biases);

        if (biases.Shape.RowCount != 1 || biases.Shape.ColumnCount != weights.Shape.ColumnCount)
        {
            throw new ArgumentException(
                $"Biases for a {weights.Shape} weight matrix must be 1x{weights.Shape.ColumnCount}. Received {biases.Shape}.",
                nameof(biases));
        }

        return new LayerParameters(weights, biases);
    }

    /// <summary>
    /// Produces the weighted input for a batch, <c>input . weights + biases</c>.
    /// </summary>
    /// <param name="input">The batch input, of shape <c>batch x inputs</c>.</param>
    /// <returns>The weighted input, of shape <c>batch x neurons</c>.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="input"/> is null.</exception>
    public Matrix WeighInput(Matrix input)
    {
        ArgumentNullException.ThrowIfNull(input);

        return input.Multiply(_weights).AddRowVector(_biases);
    }

    /// <summary>
    /// Returns a new parameter set with <paramref name="gradient"/> applied at the given step size.
    /// </summary>
    /// <param name="gradient">The gradient to descend.</param>
    /// <param name="stepSize">The learning rate to scale the gradient by.</param>
    /// <returns>The updated parameters.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="gradient"/> is null.</exception>
    public LayerParameters Descend(LayerGradient gradient, double stepSize)
    {
        ArgumentNullException.ThrowIfNull(gradient);

        return new LayerParameters(
            _weights.Subtract(gradient.Weights.Scale(stepSize)),
            _biases.Subtract(gradient.Biases.Scale(stepSize)));
    }

    /// <summary>Propagates a delta back to this layer's input, <c>delta . weights transposed</c>.</summary>
    /// <param name="delta">The delta with respect to this layer's weighted input.</param>
    /// <returns>The delta with respect to this layer's input.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="delta"/> is null.</exception>
    public Matrix PropagateDelta(Matrix delta)
    {
        ArgumentNullException.ThrowIfNull(delta);

        return delta.Multiply(_weights.Transpose());
    }
}
