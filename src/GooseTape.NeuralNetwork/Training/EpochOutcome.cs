namespace GooseTape.NeuralNetwork.Training;

/// <summary>
/// The result of one complete pass over the training data.
/// </summary>
/// <param name="Network">The network as it stood at the end of the epoch.</param>
/// <param name="Metrics">The example-weighted loss and accuracy observed across the epoch.</param>
public sealed record EpochOutcome(FeedForwardNetwork Network, TrainingMetrics Metrics);
