using GooseTape.NeuralNetwork.Activations;
using GooseTape.NeuralNetwork.Layers;
using GooseTape.NeuralNetwork.Losses;
using GooseTape.NeuralNetwork.Maths;

namespace GooseTape.NeuralNetwork.Training;

/// <summary>
/// Builds an untrained network from a topology, using He initialisation for the hidden layers.
/// </summary>
/// <remarks>
/// He initialisation scales the starting weights by <c>sqrt(2 / fanIn)</c>, which keeps the
/// variance of the signal roughly constant as it passes through rectified linear layers. Without
/// it, a 784 wide input drives early activations far enough that training stalls.
/// </remarks>
public static class NetworkInitializer
{
    /// <summary>
    /// Creates an untrained network whose hidden layers are rectified linear and whose output
    /// layer is softmax, trained against cross entropy.
    /// </summary>
    /// <param name="topology">The neuron counts, from input to output.</param>
    /// <param name="seed">The seed making initialisation reproducible.</param>
    /// <returns>A newly initialised <see cref="FeedForwardNetwork"/>.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="topology"/> is null.</exception>
    public static FeedForwardNetwork Create(NetworkTopology topology, RandomSeed seed)
    {
        ArgumentNullException.ThrowIfNull(topology);

        var sampler = GaussianSampler.From(seed);
        var layers = new DenseLayer[topology.DenseLayerCount];

        for (var index = 0; index < layers.Length; index++)
        {
            layers[index] = CreateLayer(topology, index, sampler);
        }

        return FeedForwardNetwork.Create(LayerCollection.From(layers), new CrossEntropyLoss());
    }

    private static DenseLayer CreateLayer(NetworkTopology topology, int index, GaussianSampler sampler)
    {
        var inputCount = topology[index].Value;
        var neuronCount = topology[index + 1].Value;
        var standardDeviation = Math.Sqrt(2d / inputCount);

        var parameters = LayerParameters.Create(
            Matrix.Generate(MatrixShape.Create(inputCount, neuronCount), () => sampler.NextSample(standardDeviation)),
            Matrix.Zeros(MatrixShape.Create(1, neuronCount)));

        return DenseLayer.Create(parameters, ActivationFor(topology, index));
    }

    private static IActivationFunction ActivationFor(NetworkTopology topology, int index) =>
        index == topology.DenseLayerCount - 1
            ? new SoftmaxActivation()
            : new RectifiedLinearActivation();
}
