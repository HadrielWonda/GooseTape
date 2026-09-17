using GooseTape.NeuralNetwork.Activations;
using GooseTape.NeuralNetwork.Maths;

namespace GooseTape.NeuralNetwork.Layers;

/// <summary>
/// A fully connected layer: a weighted sum of every input, followed by an activation.
/// </summary>
public sealed class DenseLayer
{
    private readonly LayerParameters _parameters;
    private readonly IActivationFunction _activation;

    private DenseLayer(LayerParameters parameters, IActivationFunction activation)
    {
        _parameters = parameters;
        _activation = activation;
    }

    /// <summary>Gets the learnable state of this layer.</summary>
    public LayerParameters Parameters => _parameters;

    /// <summary>Gets the activation applied to this layer's weighted input.</summary>
    public IActivationFunction Activation => _activation;

    /// <summary>
    /// Creates a layer from existing parameters.
    /// </summary>
    /// <param name="parameters">The weights and biases the layer starts from.</param>
    /// <param name="activation">The activation applied to the weighted input.</param>
    /// <returns>A new <see cref="DenseLayer"/>.</returns>
    /// <exception cref="ArgumentNullException">Thrown when either argument is null.</exception>
    public static DenseLayer Create(LayerParameters parameters, IActivationFunction activation)
    {
        ArgumentNullException.ThrowIfNull(parameters);
        ArgumentNullException.ThrowIfNull(activation);

        return new DenseLayer(parameters, activation);
    }

    /// <summary>
    /// Runs a batch forward through this layer.
    /// </summary>
    /// <param name="input">The batch input, of shape <c>batch x inputs</c>.</param>
    /// <returns>The input and resulting activation, retained for the backward pass.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="input"/> is null.</exception>
    public LayerForwardPass Forward(Matrix input)
    {
        ArgumentNullException.ThrowIfNull(input);

        return new LayerForwardPass(input, _activation.Activate(_parameters.WeighInput(input)));
    }

    /// <summary>
    /// Converts a delta with respect to this layer's activation into one with respect to its weighted input.
    /// </summary>
    /// <param name="activationDelta">The delta with respect to this layer's activation.</param>
    /// <param name="activation">The activation this layer produced during the matching forward pass.</param>
    /// <returns>The delta with respect to this layer's weighted input.</returns>
    /// <exception cref="ArgumentNullException">Thrown when either argument is null.</exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when this layer's activation has no standalone derivative, which means it was
    /// configured somewhere other than the output layer.
    /// </exception>
    public Matrix DeriveWeightedInputDelta(Matrix activationDelta, Matrix activation)
    {
        ArgumentNullException.ThrowIfNull(activationDelta);
        ArgumentNullException.ThrowIfNull(activation);

        return _activation switch
        {
            IDifferentiableActivation differentiable =>
                activationDelta.HadamardProduct(differentiable.DeriveFromActivation(activation)),
            _ => throw new InvalidOperationException(
                $"The '{_activation.Name}' activation has no standalone derivative and is only valid on the "
                + "output layer, where the loss function supplies the delta directly.")
        };
    }

    /// <summary>Returns a copy of this layer with <paramref name="gradient"/> applied.</summary>
    /// <param name="gradient">The gradient to descend.</param>
    /// <param name="stepSize">The learning rate to scale the gradient by.</param>
    /// <returns>A new layer holding the updated parameters.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="gradient"/> is null.</exception>
    public DenseLayer Descend(LayerGradient gradient, double stepSize)
    {
        ArgumentNullException.ThrowIfNull(gradient);

        return new DenseLayer(_parameters.Descend(gradient, stepSize), _activation);
    }
}
