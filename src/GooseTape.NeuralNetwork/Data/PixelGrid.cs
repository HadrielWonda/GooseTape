namespace GooseTape.NeuralNetwork.Data;

/// <summary>
/// The pixels of a single image, flattened to one row and normalised to <c>[0, 1]</c>.
/// </summary>
public sealed class PixelGrid
{
    /// <summary>The value an eight bit greyscale pixel is divided by to normalise it.</summary>
    private const double GreyscaleRange = 255d;

    private readonly double[] _intensities;

    private PixelGrid(double[] intensities) => _intensities = intensities;

    /// <summary>Gets the number of pixels in this grid.</summary>
    public int Count => _intensities.Length;

    /// <summary>Gets the normalised intensities, in row-major order.</summary>
    public ReadOnlySpan<double> Intensities => _intensities;

    /// <summary>
    /// Creates a grid from raw eight bit greyscale pixels, normalising them to <c>[0, 1]</c>.
    /// </summary>
    /// <param name="greyscale">The raw pixel bytes, in row-major order.</param>
    /// <returns>A new <see cref="PixelGrid"/>.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="greyscale"/> is null.</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="greyscale"/> is empty.</exception>
    public static PixelGrid FromGreyscale(ReadOnlySpan<byte> greyscale)
    {
        if (greyscale.IsEmpty)
        {
            throw new ArgumentException("An image requires at least one pixel.", nameof(greyscale));
        }

        var intensities = new double[greyscale.Length];

        for (var index = 0; index < greyscale.Length; index++)
        {
            intensities[index] = greyscale[index] / GreyscaleRange;
        }

        return new PixelGrid(intensities);
    }

    /// <summary>
    /// Creates a grid from intensities that are already normalised.
    /// </summary>
    /// <param name="intensities">The normalised intensities. Each must lie within <c>[0, 1]</c>.</param>
    /// <returns>A new <see cref="PixelGrid"/>.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="intensities"/> is null.</exception>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="intensities"/> is empty or holds a value outside <c>[0, 1]</c>.
    /// </exception>
    public static PixelGrid FromNormalised(IReadOnlyCollection<double> intensities)
    {
        ArgumentNullException.ThrowIfNull(intensities);

        if (intensities.Count == 0)
        {
            throw new ArgumentException("An image requires at least one pixel.", nameof(intensities));
        }

        if (intensities.Any(intensity => !double.IsFinite(intensity) || intensity < 0d || intensity > 1d))
        {
            throw new ArgumentException(
                "Normalised pixel intensities must all be finite values within [0, 1].",
                nameof(intensities));
        }

        return new PixelGrid([.. intensities]);
    }
}
