namespace GooseTape.NeuralNetwork.Checkpoints;

/// <summary>
/// Persists and retrieves network snapshots between epochs.
/// </summary>
/// <remarks>
/// This is the seam that makes a training run resumable. Weights are far too large to pass
/// through a portable function payload on every epoch, so the orchestrator exchanges
/// <see cref="CheckpointId"/> values while the weights themselves stay on this side of the
/// boundary.
/// </remarks>
public interface ICheckpointStore
{
    /// <summary>
    /// Stores a snapshot under <paramref name="checkpointId"/>, replacing any existing snapshot.
    /// </summary>
    /// <param name="checkpointId">The identifier to store under.</param>
    /// <param name="snapshot">The snapshot to store.</param>
    /// <param name="cancellationToken">Token to monitor for cancellation requests.</param>
    /// <returns>A task that completes once the snapshot is durable.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="snapshot"/> is null.</exception>
    Task SaveAsync(CheckpointId checkpointId, NetworkSnapshot snapshot, CancellationToken cancellationToken = default);

    /// <summary>
    /// Retrieves a previously stored snapshot.
    /// </summary>
    /// <param name="checkpointId">The identifier to retrieve.</param>
    /// <param name="cancellationToken">Token to monitor for cancellation requests.</param>
    /// <returns>The stored snapshot.</returns>
    /// <exception cref="CheckpointNotFoundException">Thrown when nothing is stored under that identifier.</exception>
    Task<NetworkSnapshot> LoadAsync(CheckpointId checkpointId, CancellationToken cancellationToken = default);

    /// <summary>
    /// Reports whether a snapshot is stored under <paramref name="checkpointId"/>.
    /// </summary>
    /// <param name="checkpointId">The identifier to test.</param>
    /// <param name="cancellationToken">Token to monitor for cancellation requests.</param>
    /// <returns><see langword="true"/> when a snapshot exists.</returns>
    Task<bool> ExistsAsync(CheckpointId checkpointId, CancellationToken cancellationToken = default);
}
