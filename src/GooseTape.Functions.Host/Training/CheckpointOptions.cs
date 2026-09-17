using System.ComponentModel.DataAnnotations;

namespace GooseTape.Functions.Host.Training;

/// <summary>
/// Where network snapshots are written between epochs.
/// </summary>
public sealed class CheckpointOptions
{
    /// <summary>The configuration section these options bind to.</summary>
    public const string SectionName = "Checkpoints";

    /// <summary>Gets or sets the directory snapshots are stored in.</summary>
    [Required(AllowEmptyStrings = false)]
    public string DirectoryPath { get; set; } = "./.checkpoints";
}
