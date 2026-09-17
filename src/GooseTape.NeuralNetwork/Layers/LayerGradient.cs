using GooseTape.NeuralNetwork.Maths;

namespace GooseTape.NeuralNetwork.Layers;

/// <summary>
/// The gradient of the loss with respect to one layer's parameters, averaged over a batch.
/// </summary>
/// <param name="Weights">The gradient for the weight matrix, matching its shape.</param>
/// <param name="Biases">The gradient for the bias row, matching its shape.</param>
public sealed record LayerGradient(Matrix Weights, Matrix Biases)
{
    /// <summary>
    /// Derives the gradient for one layer from its forward pass and the delta flowing back into it.
    /// </summary>
    /// <remarks>
    /// Depends only on what the layer saw and the delta it received, never on the layer's current
    /// weights, which is why it lives here rather than on <see cref="DenseLayer"/>.
    /// </remarks>
    /// <param name="pass">The forward pass the delta belongs to.</param>
    /// <param name="delta">The delta with respect to the layer's weighted input, of shape <c>batch x neurons</c>.</param>
    /// <returns>The batch-averaged gradient.</returns>
    /// <exception cref="ArgumentNullException">Thrown when either argument is null.</exception>
    public static LayerGradient From(LayerForwardPass pass, Matrix delta)
    {
        ArgumentNullException.ThrowIfNull(pass);
        ArgumentNullException.ThrowIfNull(delta);

        var batchScale = 1d / delta.Shape.RowCount;

        return new LayerGradient(
            pass.Input.Transpose().Multiply(delta).Scale(batchScale),
            delta.SumRows().Scale(batchScale));
    }
}
