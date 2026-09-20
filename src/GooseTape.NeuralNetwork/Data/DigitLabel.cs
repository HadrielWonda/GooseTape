namespace GooseTape.NeuralNetwork.Data;

/// <summary>
/// A handwritten digit's true class, zero through nine.
/// </summary>
/// <param name="Value">The digit.</param>
public readonly record struct DigitLabel(int Value)
{
    /// <summary>The number of distinct digit classes.</summary>
    public const int ClassCount = 10;

    /// <summary>
    /// Creates a validated digit label.
    /// </summary>
    /// <param name="value">The digit. Must lie within <c>[0, 9]</c>.</param>
    /// <returns>A validated <see cref="DigitLabel"/>.</returns>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when <paramref name="value"/> is not a digit.
    /// </exception>
    public static DigitLabel Create(int value)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(value, nameof(value));
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(value, ClassCount, nameof(value));

        return new DigitLabel(value);
    }

    /// <summary>
    /// Writes this label into <paramref name="destination"/> as a one-hot row.
    /// </summary>
    /// <param name="destination">A span of exactly <see cref="ClassCount"/> elements.</param>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="destination"/> is not <see cref="ClassCount"/> wide.
    /// </exception>
    public void WriteOneHot(Span<double> destination)
    {
        if (destination.Length != ClassCount)
        {
            throw new ArgumentException(
                $"A one-hot digit row must be {ClassCount} wide. Received {destination.Length}.",
                nameof(destination));
        }

        destination.Clear();
        destination[Value] = 1d;
    }
}
