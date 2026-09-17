using GooseTape.NeuralNetwork.Maths;

namespace GooseTape.NeuralNetwork.Activations;

/// <summary>
/// Transforms the weighted input of a layer into its activation.
/// </summary>
public interface IActivationFunction
{
    /// <summary>Gets the stable identifier persisted in checkpoints.</summary>
    string Name { get; }

    /// <summary>
    /// Applies the activation to a layer's weighted input.
    /// </summary>
    /// <param name="weightedInput">The pre-activation values, of shape <c>batch x neurons</c>.</param>
    /// <returns>The activated values, of the same shape.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="weightedInput"/> is null.</exception>
    Matrix Activate(Matrix weightedInput);
}
