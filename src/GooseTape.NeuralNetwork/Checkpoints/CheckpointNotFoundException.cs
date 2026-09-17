namespace GooseTape.NeuralNetwork.Checkpoints;

/// <summary>
/// Raised when a checkpoint is requested that was never stored, or has since been removed.
/// </summary>
public sealed class CheckpointNotFoundException : Exception
{
    /// <summary>Initialises a new instance naming the missing checkpoint.</summary>
    /// <param name="checkpointId">The checkpoint that could not be found.</param>
    public CheckpointNotFoundException(CheckpointId checkpointId)
        : base($"No checkpoint is stored under the identifier {checkpointId}.") =>
        CheckpointId = checkpointId;

    /// <summary>Initialises a new instance with a message.</summary>
    /// <param name="message">The message describing the failure.</param>
    public CheckpointNotFoundException(string message)
        : base(message) => CheckpointId = default;

    /// <summary>Initialises a new instance with a message and inner exception.</summary>
    /// <param name="message">The message describing the failure.</param>
    /// <param name="innerException">The underlying cause.</param>
    public CheckpointNotFoundException(string message, Exception innerException)
        : base(message, innerException) => CheckpointId = default;

    /// <summary>Initialises a new instance.</summary>
    public CheckpointNotFoundException()
        : base("No checkpoint is stored under the requested identifier.") => CheckpointId = default;

    /// <summary>Gets the checkpoint that could not be found.</summary>
    public CheckpointId CheckpointId { get; }
}
