using GooseTape.NeuralNetwork.Maths;

namespace GooseTape.NeuralNetwork.Data;

/// <summary>
/// A batch of examples in the matrix form the network consumes.
/// </summary>
/// <param name="Inputs">The pixel rows, of shape <c>batch x pixels</c>.</param>
/// <param name="Expected">The one-hot label rows, of shape <c>batch x classes</c>.</param>
public sealed record TrainingBatch(Matrix Inputs, Matrix Expected);
