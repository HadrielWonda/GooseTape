namespace GooseTape.NeuralNetwork.Checkpoints;

/// <summary>
/// A whole network, flattened into a form that survives serialisation.
/// </summary>
/// <param name="Layers">The layers, ordered from input side to output side.</param>
/// <param name="LossFunction">The identifier of the loss function the network trains against.</param>
/// <param name="Metadata">
/// What produced this checkpoint. Optional, because checkpoints written before lineage was
/// recorded still load; a run that needs to verify lineage treats its absence as unverifiable
/// rather than valid.
/// </param>
public sealed record NetworkSnapshot(
    IReadOnlyList<LayerSnapshot> Layers,
    string LossFunction,
    CheckpointMetadata? Metadata = null);
