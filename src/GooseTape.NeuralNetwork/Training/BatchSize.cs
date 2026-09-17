namespace GooseTape.NeuralNetwork.Training;

/// <summary>
/// The number of examples gradient descent averages over before each parameter update.
/// </summary>
/// <param name="Value">The example count.</param>
public readonly record struct BatchSize(int Value)
{
    private const int LargestSupportedBatch = 8192;

    /// <summary>
    /// Creates a validated batch size.
    /// </summary>
    /// <param name="value">The example count. Must lie within <c>[1, 8192]</c>.</param>
    /// <returns>A validated <see cref="BatchSize"/>.</returns>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when <paramref name="value"/> falls outside the supported range.
    /// </exception>
    public static BatchSize Create(int value)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(value, 1, nameof(value));
        ArgumentOutOfRangeException.ThrowIfGreaterThan(value, LargestSupportedBatch, nameof(value));

        return new BatchSize(value);
    }
}
