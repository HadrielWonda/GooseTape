namespace GooseTape.NeuralNetwork.Checkpoints;

/// <summary>
/// One dense layer, flattened into a form that survives serialisation.
/// </summary>
/// <param name="InputCount">The number of inputs the layer accepts.</param>
/// <param name="NeuronCount">The number of neurons the layer holds.</param>
/// <param name="Activation">The identifier of the layer activation.</param>
/// <param name="Weights">The weight matrix in row-major order, of length <c>InputCount * NeuronCount</c>.</param>
/// <param name="Biases">The bias row, of length <c>NeuronCount</c>.</param>
public sealed record LayerSnapshot(
    int InputCount,
    int NeuronCount,
    string Activation,
    IReadOnlyList<double> Weights,
    IReadOnlyList<double> Biases);
