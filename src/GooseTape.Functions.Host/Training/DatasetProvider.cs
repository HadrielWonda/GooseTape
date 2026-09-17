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
    private readonly Lazy<Task<ImageDataset>> _training;
    private readonly Lazy<Task<ImageDataset>> _test;

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

        _training = new Lazy<Task<ImageDataset>>(() => TrainingSource(settings).LoadAsync());
        _test = new Lazy<Task<ImageDataset>>(() => TestSource(settings).LoadAsync());
    }

    /// <summary>Gets the training dataset, loading it on first use.</summary>
    /// <returns>The training dataset.</returns>
    public Task<ImageDataset> TrainingAsync() => _training.Value;

    /// <summary>Gets the test dataset, loading it on first use.</summary>
    /// <returns>The test dataset.</returns>
    public Task<ImageDataset> TestAsync() => _test.Value;

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
}
