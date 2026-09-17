namespace GooseTape.NeuralNetwork.Training;

/// <summary>
/// The step size gradient descent applies, constrained to a range that cannot diverge outright.
/// </summary>
/// <param name="Value">The step size.</param>
public readonly record struct LearningRate(double Value)
{
    private const double SmallestUsefulRate = 1e-8;
    private const double LargestStableRate = 1d;

    /// <summary>
    /// Creates a validated learning rate.
    /// </summary>
    /// <param name="value">The step size. Must lie within <c>(0, 1]</c>.</param>
    /// <returns>A validated <see cref="LearningRate"/>.</returns>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when <paramref name="value"/> is not finite or falls outside the supported range.
    /// </exception>
    public static LearningRate Create(double value)
    {
        if (!double.IsFinite(value) || value < SmallestUsefulRate || value > LargestStableRate)
        {
            throw new ArgumentOutOfRangeException(
                nameof(value),
                value,
                $"Learning rate must be a finite value between {SmallestUsefulRate} and {LargestStableRate}.");
        }

        return new LearningRate(value);
    }
}
