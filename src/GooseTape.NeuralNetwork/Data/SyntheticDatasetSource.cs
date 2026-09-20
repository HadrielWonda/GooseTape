namespace GooseTape.NeuralNetwork.Data;

/// <summary>
/// Generates a small, deterministic, separable dataset in place of real MNIST data.
/// </summary>
/// <remarks>
/// <para>
/// Each class activates its own pair of pixels, and every pixel carries reproducible noise, so
/// the problem is learnable but not solvable from a single pixel.
/// </para>
/// <para>
/// This exists so the whole pipeline, from the durable training feature down to the checkpoint
/// store, can be exercised end to end before anyone downloads the sixty thousand image MNIST
/// archive. It is not a substitute for evaluating on real data.
/// </para>
/// </remarks>
public sealed class SyntheticDatasetSource : IImageDatasetSource
{
    private const int PixelsPerClass = 2;
    private const double SignalStrength = 0.85;
    private const double NoiseAmplitude = 0.15;

    /// <summary>The number of pixels every generated image carries.</summary>
    public const int PixelCount = DigitLabel.ClassCount * PixelsPerClass;

    private readonly int _examplesPerClass;
    private readonly int _seed;

    private SyntheticDatasetSource(int examplesPerClass, int seed)
    {
        _examplesPerClass = examplesPerClass;
        _seed = seed;
    }

    /// <summary>
    /// Creates a generator.
    /// </summary>
    /// <param name="examplesPerClass">How many examples to emit per digit. Must be positive.</param>
    /// <param name="seed">The seed making the noise reproducible.</param>
    /// <returns>A new <see cref="SyntheticDatasetSource"/>.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="examplesPerClass"/> is below one.</exception>
    public static SyntheticDatasetSource Create(int examplesPerClass, int seed)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(examplesPerClass, 1, nameof(examplesPerClass));

        return new SyntheticDatasetSource(examplesPerClass, seed);
    }

    /// <inheritdoc />
    public Task<ImageDataset> LoadAsync(CancellationToken cancellationToken = default) =>
        Task.FromResult(Build());

    /// <summary>Builds the dataset synchronously.</summary>
    /// <returns>The generated dataset.</returns>
    public ImageDataset Build()
    {
        var generator = new Random(_seed);
        var total = _examplesPerClass * DigitLabel.ClassCount;
        var images = new List<LabelledImage>(total);

        for (var index = 0; index < total; index++)
        {
            var label = DigitLabel.Create(index % DigitLabel.ClassCount);
            images.Add(LabelledImage.Create(BuildPixels(label, generator), label));
        }

        return ImageDataset.From(images);
    }

    private static PixelGrid BuildPixels(DigitLabel label, Random generator)
    {
        var pixels = new double[PixelCount];

        for (var index = 0; index < pixels.Length; index++)
        {
            pixels[index] = generator.NextDouble() * NoiseAmplitude;
        }

        var signalStart = label.Value * PixelsPerClass;
        pixels[signalStart] = Math.Clamp(pixels[signalStart] + SignalStrength, 0d, 1d);
        pixels[signalStart + 1] = Math.Clamp(pixels[signalStart + 1] + SignalStrength, 0d, 1d);

        return PixelGrid.FromNormalised(pixels);
    }
}
