using GooseTape.NeuralNetwork.Maths;

namespace GooseTape.NeuralNetwork.Activations;

/// <summary>
/// The rectified linear unit, <c>max(0, x)</c>.
/// </summary>
public sealed class RectifiedLinearActivation : IDifferentiableActivation
{
    /// <summary>The identifier persisted in checkpoints for this activation.</summary>
    public const string Identifier = "relu";

    /// <inheritdoc />
    public string Name => Identifier;

    /// <inheritdoc />
    public Matrix Activate(Matrix weightedInput)
    {
        ArgumentNullException.ThrowIfNull(weightedInput);

        return weightedInput.Map(static value => Math.Max(0d, value));
    }

    /// <inheritdoc />
    public Matrix DeriveFromActivation(Matrix activation)
    {
        ArgumentNullException.ThrowIfNull(activation);

        return activation.Map(static value => value > 0d ? 1d : 0d);
    }
}
