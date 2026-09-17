using GooseTape.NeuralNetwork.Maths;

namespace GooseTape.NeuralNetwork.Losses;

/// <summary>
/// Categorical cross entropy, paired with a softmax output layer.
/// </summary>
/// <remarks>
/// <see cref="DeriveOutputDelta"/> returns <c>prediction - expected</c>. That is the gradient of
/// cross entropy with respect to the output layer's weighted input once the softmax Jacobian is
/// folded in, which is why softmax never needs to expose a derivative of its own.
/// </remarks>
public sealed class CrossEntropyLoss : ILossFunction
{
    /// <summary>The identifier persisted in checkpoints for this loss function.</summary>
    public const string Identifier = "cross-entropy";

    /// <summary>
    /// Guards against taking the logarithm of zero when the network is fully confident and wrong.
    /// </summary>
    private const double ProbabilityFloor = 1e-12;

    /// <inheritdoc />
    public string Name => Identifier;

    /// <inheritdoc />
    public double Compute(Matrix prediction, Matrix expected)
    {
        ArgumentNullException.ThrowIfNull(prediction);
        ArgumentNullException.ThrowIfNull(expected);

        var perExample = expected
            .HadamardProduct(prediction.Map(static value => -Math.Log(Math.Max(value, ProbabilityFloor))))
            .Sum();

        return perExample / prediction.Shape.RowCount;
    }

    /// <inheritdoc />
    public Matrix DeriveOutputDelta(Matrix prediction, Matrix expected)
    {
        ArgumentNullException.ThrowIfNull(prediction);
        ArgumentNullException.ThrowIfNull(expected);

        return prediction.Subtract(expected);
    }
}
