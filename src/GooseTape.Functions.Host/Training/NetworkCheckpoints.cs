using GooseTape.Functions.Host.Ductape;
using GooseTape.NeuralNetwork.Checkpoints;
using GooseTape.NeuralNetwork.Training;

namespace GooseTape.Functions.Host.Training;

/// <summary>
/// Loads and stores networks by checkpoint identifier, translating storage failures into the
/// error codes Ductape understands.
/// </summary>
public sealed class NetworkCheckpoints
{
    private readonly ICheckpointStore _store;
    private readonly NetworkSerializer _serializer;

    /// <summary>
    /// Creates the service over a store and serialiser.
    /// </summary>
    /// <param name="store">Where snapshots are persisted.</param>
    /// <param name="serializer">Converts networks to and from snapshots.</param>
    /// <exception cref="ArgumentNullException">Thrown when either argument is null.</exception>
    public NetworkCheckpoints(ICheckpointStore store, NetworkSerializer serializer)
    {
        ArgumentNullException.ThrowIfNull(store);
        ArgumentNullException.ThrowIfNull(serializer);

        _store = store;
        _serializer = serializer;
    }

    /// <summary>
    /// Parses and validates a checkpoint identifier supplied as function input.
    /// </summary>
    /// <param name="value">The raw identifier.</param>
    /// <returns>The validated identifier.</returns>
    /// <exception cref="PortableFunctionException">Thrown when the identifier is not well formed.</exception>
    public static CheckpointId Parse(string value)
    {
        try
        {
            return CheckpointId.Create(value);
        }
        catch (ArgumentException failure)
        {
            throw new PortableFunctionException("FUNCTION_INPUT_INVALID", failure.Message);
        }
    }

    /// <summary>
    /// Loads a network from a checkpoint.
    /// </summary>
    /// <param name="checkpointId">The checkpoint to load.</param>
    /// <param name="cancellationToken">Token to monitor for cancellation requests.</param>
    /// <returns>The restored network.</returns>
    /// <exception cref="PortableFunctionException">
    /// Thrown when the checkpoint is missing, or holds a snapshot this host cannot rebuild.
    /// </exception>
    public async Task<FeedForwardNetwork> LoadAsync(CheckpointId checkpointId, CancellationToken cancellationToken)
    {
        try
        {
            var snapshot = await _store.LoadAsync(checkpointId, cancellationToken).ConfigureAwait(false);

            return _serializer.FromSnapshot(snapshot);
        }
        catch (CheckpointNotFoundException failure)
        {
            throw new PortableFunctionException("CHECKPOINT_NOT_FOUND", failure.Message);
        }
        catch (InvalidDataException failure)
        {
            throw new PortableFunctionException("CHECKPOINT_CORRUPT", failure.Message);
        }
        catch (NotSupportedException failure)
        {
            throw new PortableFunctionException("CHECKPOINT_UNSUPPORTED", failure.Message);
        }
    }

    /// <summary>
    /// Stores a network under a checkpoint identifier.
    /// </summary>
    /// <param name="checkpointId">The identifier to store under.</param>
    /// <param name="network">The network to store.</param>
    /// <param name="cancellationToken">Token to monitor for cancellation requests.</param>
    /// <returns>A task that completes once the snapshot is durable.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="network"/> is null.</exception>
    /// <exception cref="PortableFunctionException">Thrown when the snapshot cannot be written.</exception>
    public async Task SaveAsync(
        CheckpointId checkpointId,
        FeedForwardNetwork network,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(network);

        try
        {
            await _store
                .SaveAsync(checkpointId, NetworkSerializer.ToSnapshot(network), cancellationToken)
                .ConfigureAwait(false);
        }
        catch (IOException failure)
        {
            // Storage problems are usually transient, so Ductape is told the step may be retried.
            throw new PortableFunctionException("CHECKPOINT_WRITE_FAILED", failure.Message, retryable: true);
        }
    }
}
