namespace GooseTape.NeuralNetwork.Training;

/// <summary>
/// What one epoch or evaluation produced.
/// </summary>
/// <param name="MeanLoss">The mean loss across every example considered.</param>
/// <param name="Accuracy">The fraction of examples classified correctly, within <c>[0, 1]</c>.</param>
/// <param name="ExampleCount">The number of examples the metrics were computed over.</param>
public readonly record struct TrainingMetrics(double MeanLoss, double Accuracy, int ExampleCount)
{
    /// <summary>
    /// Creates a validated metric set.
    /// </summary>
    /// <param name="meanLoss">The mean loss. Must be finite and non-negative.</param>
    /// <param name="accuracy">The accuracy. Must lie within <c>[0, 1]</c>.</param>
    /// <param name="exampleCount">The number of examples measured. Must be positive.</param>
    /// <returns>A validated <see cref="TrainingMetrics"/>.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when any argument falls outside its range.</exception>
    public static TrainingMetrics Create(double meanLoss, double accuracy, int exampleCount)
    {
        if (!double.IsFinite(meanLoss) || meanLoss < 0d)
        {
            throw new ArgumentOutOfRangeException(
                nameof(meanLoss), meanLoss, "Mean loss must be a finite, non-negative value.");
        }

        if (!double.IsFinite(accuracy) || accuracy < 0d || accuracy > 1d)
        {
            throw new ArgumentOutOfRangeException(
                nameof(accuracy), accuracy, "Accuracy must be a finite value within [0, 1].");
        }

        ArgumentOutOfRangeException.ThrowIfLessThan(exampleCount, 1, nameof(exampleCount));

        return new TrainingMetrics(meanLoss, accuracy, exampleCount);
    }
}
