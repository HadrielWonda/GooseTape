namespace GooseTape.NeuralNetwork.Training;

/// <summary>
/// A one based position in a training schedule.
/// </summary>
/// <param name="Value">The epoch position, starting at one.</param>
public readonly record struct EpochNumber(int Value)
{
    /// <summary>
    /// Creates a validated epoch number.
    /// </summary>
    /// <param name="value">The epoch position. Must be one or greater.</param>
    /// <returns>A validated <see cref="EpochNumber"/>.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="value"/> is below one.</exception>
    public static EpochNumber Create(int value)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(value, 1, nameof(value));

        return new EpochNumber(value);
    }

    /// <summary>Returns the epoch that follows this one.</summary>
    /// <returns>The next <see cref="EpochNumber"/>.</returns>
    public EpochNumber Next() => new(Value + 1);
}
