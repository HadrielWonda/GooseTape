using System.Collections;
using GooseTape.NeuralNetwork.Maths;
using GooseTape.NeuralNetwork.Training;

namespace GooseTape.NeuralNetwork.Data;

/// <summary>
/// A set of labelled images that can be shuffled and split into batches.
/// </summary>
public sealed class ImageDataset : IReadOnlyList<LabelledImage>
{
    private readonly IReadOnlyList<LabelledImage> _images;

    private ImageDataset(IReadOnlyList<LabelledImage> images) => _images = images;

    /// <summary>Gets the number of images in this dataset.</summary>
    public int Count => _images.Count;

    /// <summary>Gets the image at <paramref name="index"/>.</summary>
    /// <param name="index">The zero based position.</param>
    /// <returns>The image at that position.</returns>
    public LabelledImage this[int index] => _images[index];

    /// <summary>Gets the number of pixels every image in this dataset carries.</summary>
    public int PixelCount => _images[0].Pixels.Count;

    /// <summary>
    /// Creates a validated dataset.
    /// </summary>
    /// <param name="images">The images. Must be non-empty and uniformly sized.</param>
    /// <returns>A new <see cref="ImageDataset"/>.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="images"/> is null.</exception>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="images"/> is empty or the images differ in pixel count.
    /// </exception>
    public static ImageDataset From(IReadOnlyList<LabelledImage> images)
    {
        ArgumentNullException.ThrowIfNull(images);

        if (images.Count == 0)
        {
            throw new ArgumentException("A dataset requires at least one image.", nameof(images));
        }

        var pixelCount = images[0].Pixels.Count;

        if (images.Any(image => image.Pixels.Count != pixelCount))
        {
            throw new ArgumentException(
                $"Every image in a dataset must carry {pixelCount} pixels.",
                nameof(images));
        }

        return new ImageDataset([.. images]);
    }

    /// <summary>
    /// Returns a reordered copy of this dataset.
    /// </summary>
    /// <remarks>
    /// Shuffling between epochs stops gradient descent from following the same path through the
    /// data every pass, which is what keeps successive epochs from correlating.
    /// </remarks>
    /// <param name="seed">The seed controlling the permutation.</param>
    /// <returns>A new, reordered <see cref="ImageDataset"/>.</returns>
    public ImageDataset Shuffle(RandomSeed seed)
    {
        var shuffled = _images.ToArray();
        seed.ToGenerator().Shuffle(shuffled);

        return new ImageDataset(shuffled);
    }

    /// <summary>
    /// Returns a copy holding at most <paramref name="limit"/> images from the front.
    /// </summary>
    /// <param name="limit">The greatest number of images to retain. Must be positive.</param>
    /// <returns>A new <see cref="ImageDataset"/>.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when <paramref name="limit"/> is below one.</exception>
    public ImageDataset TakeAtMost(int limit)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(limit, 1, nameof(limit));

        return new ImageDataset([.. _images.Take(limit)]);
    }

    /// <summary>
    /// Splits this dataset into consecutive batches.
    /// </summary>
    /// <remarks>The final batch is smaller than the rest when the count does not divide evenly.</remarks>
    /// <param name="batchSize">The greatest number of examples per batch.</param>
    /// <returns>The batches, in order.</returns>
    public IEnumerable<TrainingBatch> Batches(BatchSize batchSize)
    {
        for (var start = 0; start < _images.Count; start += batchSize.Value)
        {
            yield return BatchFrom(start, Math.Min(batchSize.Value, _images.Count - start));
        }
    }

    /// <summary>Materialises the whole dataset as a single batch, for evaluation.</summary>
    /// <returns>One batch holding every image.</returns>
    public TrainingBatch AsSingleBatch() => BatchFrom(0, _images.Count);

    private TrainingBatch BatchFrom(int start, int length)
    {
        var pixelCount = PixelCount;
        var inputs = new double[length * pixelCount];
        var expected = new double[length * DigitLabel.ClassCount];

        for (var offset = 0; offset < length; offset++)
        {
            WriteExample(_images[start + offset], inputs.AsSpan(offset * pixelCount, pixelCount),
                expected.AsSpan(offset * DigitLabel.ClassCount, DigitLabel.ClassCount));
        }

        return new TrainingBatch(
            Matrix.Wrap(MatrixShape.Create(length, pixelCount), inputs),
            Matrix.Wrap(MatrixShape.Create(length, DigitLabel.ClassCount), expected));
    }

    private static void WriteExample(LabelledImage image, Span<double> inputRow, Span<double> expectedRow)
    {
        image.Pixels.Intensities.CopyTo(inputRow);
        image.Label.WriteOneHot(expectedRow);
    }

    /// <inheritdoc />
    public IEnumerator<LabelledImage> GetEnumerator() => _images.GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}
