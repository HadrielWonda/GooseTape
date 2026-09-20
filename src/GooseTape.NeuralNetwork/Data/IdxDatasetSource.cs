namespace GooseTape.NeuralNetwork.Data;

/// <summary>
/// Loads a dataset from a matched pair of MNIST IDX files.
/// </summary>
public sealed class IdxDatasetSource : IImageDatasetSource
{
    private readonly string _imageFilePath;
    private readonly string _labelFilePath;

    private IdxDatasetSource(string imageFilePath, string labelFilePath)
    {
        _imageFilePath = imageFilePath;
        _labelFilePath = labelFilePath;
    }

    /// <summary>
    /// Creates a source over an image file and its matching label file.
    /// </summary>
    /// <param name="imageFilePath">The IDX image file, optionally gzip compressed.</param>
    /// <param name="labelFilePath">The IDX label file, optionally gzip compressed.</param>
    /// <returns>A new <see cref="IdxDatasetSource"/>.</returns>
    /// <exception cref="ArgumentException">Thrown when either path is null or blank.</exception>
    public static IdxDatasetSource Create(string imageFilePath, string labelFilePath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(imageFilePath);
        ArgumentException.ThrowIfNullOrWhiteSpace(labelFilePath);

        return new IdxDatasetSource(imageFilePath, labelFilePath);
    }

    /// <inheritdoc />
    public async Task<ImageDataset> LoadAsync(CancellationToken cancellationToken = default)
    {
        var images = await IdxFileReader.ReadImagesAsync(_imageFilePath, cancellationToken).ConfigureAwait(false);
        var labels = await IdxFileReader.ReadLabelsAsync(_labelFilePath, cancellationToken).ConfigureAwait(false);

        if (images.Count != labels.Count)
        {
            throw new InvalidDataException(
                $"{_imageFilePath} holds {images.Count} images but {_labelFilePath} holds {labels.Count} labels.");
        }

        return ImageDataset.From([.. images.Zip(labels, LabelledImage.Create)]);
    }
}
