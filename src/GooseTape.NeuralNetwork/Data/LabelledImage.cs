namespace GooseTape.NeuralNetwork.Data;

/// <summary>
/// One training or test example: an image paired with the digit it depicts.
/// </summary>
public sealed class LabelledImage
{
    private readonly PixelGrid _pixels;
    private readonly DigitLabel _label;

    private LabelledImage(PixelGrid pixels, DigitLabel label)
    {
        _pixels = pixels;
        _label = label;
    }

    /// <summary>Gets the normalised pixels of this image.</summary>
    public PixelGrid Pixels => _pixels;

    /// <summary>Gets the digit this image depicts.</summary>
    public DigitLabel Label => _label;

    /// <summary>
    /// Pairs an image with its label.
    /// </summary>
    /// <param name="pixels">The normalised pixels.</param>
    /// <param name="label">The digit depicted.</param>
    /// <returns>A new <see cref="LabelledImage"/>.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="pixels"/> is null.</exception>
    public static LabelledImage Create(PixelGrid pixels, DigitLabel label)
    {
        ArgumentNullException.ThrowIfNull(pixels);

        return new LabelledImage(pixels, label);
    }
}
