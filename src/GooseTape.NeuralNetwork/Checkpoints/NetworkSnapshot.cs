namespace GooseTape.NeuralNetwork.Checkpoints;

/// <summary>
/// A whole network, flattened into a form that survives serialisation.
/// </summary>
/// <param name="Layers">The layers, ordered from input side to output side.</param>
/// <param name="LossFunction">The identifier of the loss function the network trains against.</param>
public sealed record NetworkSnapshot(IReadOnlyList<LayerSnapshot> Layers, string LossFunction);
