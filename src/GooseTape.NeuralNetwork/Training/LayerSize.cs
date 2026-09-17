namespace GooseTape.NeuralNetwork.Training;

/// <summary>
/// The number of neurons in one layer of a topology.
/// </summary>
/// <param name="Value">The neuron count.</param>
public readonly record struct LayerSize(int Value)
{
    private const int LargestSupportedLayer = 65536;

    /// <summary>
    /// Creates a validated layer size.
    /// </summary>
    /// <param name="value">The neuron count. Must lie within <c>[1, 65536]</c>.</param>
    /// <returns>A validated <see cref="LayerSize"/>.</returns>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when <paramref name="value"/> falls outside the supported range.
    /// </exception>
    public static LayerSize Create(int value)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(value, 1, nameof(value));
        ArgumentOutOfRangeException.ThrowIfGreaterThan(value, LargestSupportedLayer, nameof(value));

        return new LayerSize(value);
    }
}
