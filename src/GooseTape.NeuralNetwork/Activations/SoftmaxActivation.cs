using GooseTape.NeuralNetwork.Maths;

namespace GooseTape.NeuralNetwork.Activations;

/// <summary>
/// Row-wise softmax, turning each row of logits into a probability distribution.
/// </summary>
/// <remarks>
/// Each row has its maximum subtracted before exponentiation. Without that shift the
/// exponential overflows for logits beyond roughly 709, which is reachable early in training
/// when weights are poorly scaled.
/// </remarks>
public sealed class SoftmaxActivation : IActivationFunction
{
    /// <summary>The identifier persisted in checkpoints for this activation.</summary>
    public const string Identifier = "softmax";

    /// <inheritdoc />
    public string Name => Identifier;

    /// <inheritdoc />
    public Matrix Activate(Matrix weightedInput)
    {
        ArgumentNullException.ThrowIfNull(weightedInput);

        var values = weightedInput.ToArray();
        var columnCount = weightedInput.Shape.ColumnCount;

        for (var row = 0; row < weightedInput.Shape.RowCount; row++)
        {
            NormaliseRow(values.AsSpan(row * columnCount, columnCount));
        }

        return Matrix.Wrap(weightedInput.Shape, values);
    }

    private static void NormaliseRow(Span<double> row)
    {
        var largest = LargestOf(row);
        var total = 0d;

        for (var index = 0; index < row.Length; index++)
        {
            row[index] = Math.Exp(row[index] - largest);
            total += row[index];
        }

        DivideBy(row, total);
    }

    private static double LargestOf(ReadOnlySpan<double> row)
    {
        var largest = double.NegativeInfinity;

        foreach (var value in row)
        {
            largest = Math.Max(largest, value);
        }

        return largest;
    }

    private static void DivideBy(Span<double> row, double total)
    {
        for (var index = 0; index < row.Length; index++)
        {
            row[index] /= total;
        }
    }
}
