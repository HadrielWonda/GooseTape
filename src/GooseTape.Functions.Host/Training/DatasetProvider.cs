using System.Security.Cryptography;
using GooseTape.NeuralNetwork.Data;
using Microsoft.Extensions.Options;

namespace GooseTape.Functions.Host.Training;

/// <summary>
/// Loads the training and test datasets once per process and hands the same instances to every
/// subsequent invocation.
/// </summary>
/// <remarks>
/// <para>
/// A durable training run calls this host once per epoch. Re-reading and re-decoding sixty
/// thousand images on every one of those calls would dominate the cost of training, so the
/// datasets are loaded lazily and then held for the lifetime of the process.
/// </para>
/// <para>
/// The cancellation token of whichever invocation triggers the first load governs that load.
/// Later callers await the same task and are not able to cancel it, which is the usual trade
/// for sharing one in-flight load between concurrent callers.
/// </para>
/// </remarks>
public sealed class DatasetProvider
{
    /// <summary>How many characters of the content hash identify an IDX dataset.</summary>
    private const int DatasetIdLength = 16;

    private readonly DatasetHandles _handles;

    /// <summary>
    /// Creates a provider over the configured data sources.
    /// </summary>
    /// <param name="options">The configured dataset options.</param>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="options"/> is null.</exception>
    /// <exception cref="OptionsValidationException">Thrown when the configuration is incomplete.</exception>
    public DatasetProvider(IOptions<DatasetOptions> options)
    {
        ArgumentNullException.ThrowIfNull(options);

        var settings = options.Value;

        _handles = new DatasetHandles(
            new Lazy<Task<ImageDataset>>(() => TrainingSource(settings).LoadAsync()),
            new Lazy<Task<ImageDataset>>(() => TestSource(settings).LoadAsync()),
            new Lazy<Task<string>>(() => IdentifyAsync(settings)));
    }

    /// <summary>Gets the training dataset, loading it on first use.</summary>
    /// <returns>The training dataset.</returns>
    public Task<ImageDataset> TrainingAsync() => _handles.Training.Value;

    /// <summary>Gets the test dataset, loading it on first use.</summary>
    /// <returns>The test dataset.</returns>
    public Task<ImageDataset> TestAsync() => _handles.Test.Value;

    /// <summary>
    /// Gets a stable identifier for the training data.
    /// </summary>
    /// <remarks>
    /// Real data is identified by the content of its image and label files, so swapping the
    /// dataset underneath a run changes the identifier even when the paths do not. Generated data
    /// is identified by the parameters that produce it, which determine it completely.
    /// </remarks>
    /// <returns>The dataset identifier.</returns>
    public Task<string> DatasetIdAsync() => _handles.Identity.Value;

    private static async Task<string> IdentifyAsync(DatasetOptions settings)
    {
        if (!settings.UsesIdxFiles)
        {
            return $"synthetic:{settings.SyntheticTrainingExamplesPerClass}:{settings.SyntheticSeed}";
        }

        var images = RequirePath(settings.TrainingImagesPath, nameof(DatasetOptions.TrainingImagesPath));
        var labels = RequirePath(settings.TrainingLabelsPath, nameof(DatasetOptions.TrainingLabelsPath));

        return $"idx:{await HashFilesAsync(images, labels).ConfigureAwait(false)}";
    }

    private static async Task<string> HashFilesAsync(string imagePath, string labelPath)
    {
        using var hash = SHA256.Create();

        foreach (var path in new[] { imagePath, labelPath })
        {
            await using var file = File.OpenRead(path);
            var buffer = new byte[81920];
            int read;

            while ((read = await file.ReadAsync(buffer).ConfigureAwait(false)) > 0)
            {
                hash.TransformBlock(buffer, 0, read, null, 0);
            }
        }

        hash.TransformFinalBlock([], 0, 0);

        return Convert.ToHexStringLower(hash.Hash!)[..DatasetIdLength];
    }

    private static IImageDatasetSource TrainingSource(DatasetOptions settings) => settings.UsesIdxFiles
        ? IdxDatasetSource.Create(
            RequirePath(settings.TrainingImagesPath, nameof(DatasetOptions.TrainingImagesPath)),
            RequirePath(settings.TrainingLabelsPath, nameof(DatasetOptions.TrainingLabelsPath)))
        : SyntheticDatasetSource.Create(settings.SyntheticTrainingExamplesPerClass, settings.SyntheticSeed);

    private static IImageDatasetSource TestSource(DatasetOptions settings) => settings.UsesIdxFiles
        ? IdxDatasetSource.Create(
            RequirePath(settings.TestImagesPath, nameof(DatasetOptions.TestImagesPath)),
            RequirePath(settings.TestLabelsPath, nameof(DatasetOptions.TestLabelsPath)))
        : SyntheticDatasetSource.Create(settings.SyntheticTestExamplesPerClass, settings.SyntheticSeed + 1);

    private static string RequirePath(string? value, string settingName)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            throw new InvalidOperationException(
                $"Dataset:{settingName} must be configured when Dataset:Provider is {DatasetOptions.IdxProvider}.");
        }

        return value;
    }

    private sealed record DatasetHandles(
        Lazy<Task<ImageDataset>> Training,
        Lazy<Task<ImageDataset>> Test,
        Lazy<Task<string>> Identity);
}
