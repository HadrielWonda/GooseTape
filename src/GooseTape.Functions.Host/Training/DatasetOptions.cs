using System.ComponentModel.DataAnnotations;

namespace GooseTape.Functions.Host.Training;

/// <summary>
/// Where the host reads its training and test data from.
/// </summary>
/// <remarks>
/// The synthetic provider exists so the pipeline can be run end to end without the MNIST
/// download. Switch to <c>idx</c> and supply the four file paths to train on real digits.
/// </remarks>
public sealed class DatasetOptions
{
    /// <summary>The configuration section these options bind to.</summary>
    public const string SectionName = "Dataset";

    /// <summary>The provider name selecting real IDX files.</summary>
    public const string IdxProvider = "idx";

    /// <summary>The provider name selecting generated data.</summary>
    public const string SyntheticProvider = "synthetic";

    /// <summary>Gets or sets the provider, either <c>idx</c> or <c>synthetic</c>.</summary>
    [Required]
    [RegularExpression($"^({IdxProvider}|{SyntheticProvider})$")]
    public string Provider { get; set; } = SyntheticProvider;

    /// <summary>Gets or sets the IDX file holding the training images.</summary>
    public string? TrainingImagesPath { get; set; }

    /// <summary>Gets or sets the IDX file holding the training labels.</summary>
    public string? TrainingLabelsPath { get; set; }

    /// <summary>Gets or sets the IDX file holding the test images.</summary>
    public string? TestImagesPath { get; set; }

    /// <summary>Gets or sets the IDX file holding the test labels.</summary>
    public string? TestLabelsPath { get; set; }

    /// <summary>Gets or sets how many examples per digit the synthetic provider generates for training.</summary>
    [Range(1, 100_000)]
    public int SyntheticTrainingExamplesPerClass { get; set; } = 240;

    /// <summary>Gets or sets how many examples per digit the synthetic provider generates for testing.</summary>
    [Range(1, 100_000)]
    public int SyntheticTestExamplesPerClass { get; set; } = 60;

    /// <summary>Gets or sets the seed the synthetic provider generates from.</summary>
    public int SyntheticSeed { get; set; } = 20_260_917;

    /// <summary>Gets whether this configuration selects the real IDX provider.</summary>
    public bool UsesIdxFiles => string.Equals(Provider, IdxProvider, StringComparison.OrdinalIgnoreCase);
}
