using GooseTape.NeuralNetwork.Maths;

namespace GooseTape.NeuralNetwork.Activations;

/// <summary>
/// An activation whose derivative can be evaluated independently of the loss function.
/// </summary>
/// <remarks>
/// Softmax is deliberately excluded. Its Jacobian is only tractable in combination with
/// cross entropy, so that pairing is expressed by <c>ILossFunction.DeriveOutputDelta</c>
/// instead of by this interface. Segregating the two keeps every implementer of this
/// interface fully substitutable.
/// </remarks>
public interface IDifferentiableActivation : IActivationFunction
{
    /// <summary>
    /// Evaluates the derivative with respect to the weighted input, given the already computed activation.
    /// </summary>
    /// <param name="activation">The output previously produced by <see cref="IActivationFunction.Activate"/>.</param>
    /// <returns>The element-wise derivative, of the same shape.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="activation"/> is null.</exception>
    Matrix DeriveFromActivation(Matrix activation);
}
