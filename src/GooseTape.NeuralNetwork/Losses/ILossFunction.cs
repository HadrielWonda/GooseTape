using GooseTape.NeuralNetwork.Maths;

namespace GooseTape.NeuralNetwork.Losses;

/// <summary>
/// Scores how far a prediction sits from the expected distribution, and supplies the
/// gradient the backward pass starts from.
/// </summary>
public interface ILossFunction
{
    /// <summary>Gets the stable identifier persisted in checkpoints.</summary>
    string Name { get; }

    /// <summary>
    /// Computes the mean loss across every row in the batch.
    /// </summary>
    /// <param name="prediction">The predicted distributions, of shape <c>batch x classes</c>.</param>
    /// <param name="expected">The one-hot expected distributions, of the same shape.</param>
    /// <returns>The mean loss over the batch.</returns>
    /// <exception cref="ArgumentNullException">Thrown when either argument is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the shapes differ.</exception>
    double Compute(Matrix prediction, Matrix expected);

    /// <summary>
    /// Computes the gradient of the loss with respect to the output layer's weighted input.
    /// </summary>
    /// <param name="prediction">The predicted distributions, of shape <c>batch x classes</c>.</param>
    /// <param name="expected">The one-hot expected distributions, of the same shape.</param>
    /// <returns>The delta the backward pass propagates, of the same shape.</returns>
    /// <exception cref="ArgumentNullException">Thrown when either argument is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the shapes differ.</exception>
    Matrix DeriveOutputDelta(Matrix prediction, Matrix expected);
}
