namespace GooseTape.NeuralNetwork.Training;

/// <summary>
/// Combines per batch metrics into the totals for a whole epoch.
/// </summary>
/// <remarks>
/// Batches are weighted by their example count rather than averaged evenly, because the final
/// batch of an epoch is usually smaller than the rest and would otherwise be over-counted.
/// </remarks>
public sealed class TrainingMetricsAccumulator
{
    private MetricTotals _totals;

    /// <summary>Creates an empty accumulator.</summary>
    public TrainingMetricsAccumulator() => _totals = new MetricTotals(0d, 0d, 0);

    /// <summary>
    /// Folds one batch of metrics into the running totals.
    /// </summary>
    /// <param name="metrics">The metrics measured for a single batch.</param>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when the metrics are not well formed.</exception>
    public void Add(TrainingMetrics metrics) => _totals = new MetricTotals(
        _totals.WeightedLoss + (metrics.MeanLoss * metrics.ExampleCount),
        _totals.WeightedCorrect + (metrics.Accuracy * metrics.ExampleCount),
        _totals.ExampleCount + metrics.ExampleCount);

    /// <summary>
    /// Produces the combined metrics.
    /// </summary>
    /// <returns>The example-weighted mean loss and accuracy.</returns>
    /// <exception cref="InvalidOperationException">Thrown when nothing has been accumulated.</exception>
    public TrainingMetrics ToMetrics()
    {
        if (_totals.ExampleCount == 0)
        {
            throw new InvalidOperationException("No batches have been accumulated, so there are no metrics to report.");
        }

        return TrainingMetrics.Create(
            _totals.WeightedLoss / _totals.ExampleCount,
            Math.Clamp(_totals.WeightedCorrect / _totals.ExampleCount, 0d, 1d),
            _totals.ExampleCount);
    }

    private readonly record struct MetricTotals(double WeightedLoss, double WeightedCorrect, int ExampleCount);
}
