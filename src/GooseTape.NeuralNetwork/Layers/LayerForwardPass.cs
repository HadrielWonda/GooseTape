using GooseTape.NeuralNetwork.Maths;

namespace GooseTape.NeuralNetwork.Layers;

/// <summary>
/// What a layer saw and produced during one forward pass, retained so the backward pass
/// can compute gradients without recomputing the forward direction.
/// </summary>
/// <param name="Input">The batch the layer received, of shape <c>batch x inputs</c>.</param>
/// <param name="Activation">The batch the layer emitted, of shape <c>batch x neurons</c>.</param>
public sealed record LayerForwardPass(Matrix Input, Matrix Activation);
