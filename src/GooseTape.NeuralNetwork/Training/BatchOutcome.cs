namespace GooseTape.NeuralNetwork.Training;

/// <summary>
/// The result of training on a single batch: the updated network and what it cost.
/// </summary>
/// <param name="Network">The network after the parameter update.</param>
/// <param name="Metrics">
/// The loss and accuracy measured on the forward pass that produced the update, so before it was
/// applied. Reusing that pass is what keeps an epoch to one traversal of the data.
/// </param>
public sealed record BatchOutcome(FeedForwardNetwork Network, TrainingMetrics Metrics);
