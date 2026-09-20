namespace GooseTape.NeuralNetwork.Checkpoints;

/// <summary>
/// What a checkpoint is, and what produced it.
/// </summary>
/// <remarks>
/// <para>
/// Without this, a checkpoint is only weights, and everything else about it lives in its file
/// name. That is enough to load a network but not enough to trust one: you cannot tell which run
/// wrote it, which checkpoint it continued from, or which data it was trained on. Recording the
/// lineage lets a training step verify that the checkpoint it was handed is the one it expects,
/// rather than assuming the name is honest.
/// </para>
/// <para>
/// <paramref name="LearningRate"/> and <paramref name="BatchSize"/> are absent on the epoch zero
/// checkpoint, which is initialised rather than trained.
/// </para>
/// </remarks>
/// <param name="RunId">The training run that wrote this checkpoint.</param>
/// <param name="Epoch">The epoch this checkpoint holds the result of. Zero means freshly initialised.</param>
/// <param name="ParentCheckpointId">The checkpoint this one continued from, absent at epoch zero.</param>
/// <param name="DatasetId">Identifies the data the run trained on.</param>
/// <param name="Topology">The layer sizes, such as <c>784-128-10</c>.</param>
/// <param name="Seed">The run seed, which together with the epoch fixes the shuffle order.</param>
/// <param name="LearningRate">The step size the epoch used.</param>
/// <param name="BatchSize">The batch size the epoch used.</param>
/// <param name="CreatedAt">When the checkpoint was written.</param>
/// <param name="FormatVersion">The snapshot format, so a future change can be detected rather than guessed at.</param>
public sealed record CheckpointMetadata(
    string RunId,
    int Epoch,
    string? ParentCheckpointId,
    string DatasetId,
    string Topology,
    int Seed,
    double? LearningRate,
    int? BatchSize,
    DateTimeOffset CreatedAt,
    int FormatVersion)
{
    /// <summary>The snapshot format this library writes.</summary>
    public const int CurrentFormatVersion = 1;

    /// <summary>
    /// Describes a topology the way <see cref="Topology"/> records it.
    /// </summary>
    /// <param name="layerSizes">The layer sizes, from input to output.</param>
    /// <returns>The sizes joined by hyphens.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="layerSizes"/> is null.</exception>
    public static string DescribeTopology(IEnumerable<int> layerSizes)
    {
        ArgumentNullException.ThrowIfNull(layerSizes);

        return string.Join('-', layerSizes);
    }
}
