using GooseTape.NeuralNetwork.Checkpoints;
using GooseTape.NeuralNetwork.Maths;
using GooseTape.NeuralNetwork.Training;

namespace GooseTape.NeuralNetwork.Tests;

/// <summary>
/// Checks the analytical gradients backpropagation produces against gradients estimated
/// numerically by finite differences. This is the test that would catch a transposed matrix or
/// a dropped activation derivative, which otherwise show up only as a network that trains
/// slightly worse than it should.
/// </summary>
public sealed class BackpropagationGradientTests
{
    private const double Perturbation = 1e-6;
    private const double Tolerance = 1e-6;

    private static readonly Matrix Inputs =
        Matrix.FromValues(MatrixShape.Create(3, 3), [0.10, 0.60, 0.30, 0.90, 0.20, 0.40, 0.50, 0.50, 0.70]);

    private static readonly Matrix Expected =
        Matrix.FromValues(MatrixShape.Create(3, 2), [1, 0, 0, 1, 1, 0]);

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void WeightGradients_MatchNumericalEstimates(int layerIndex)
    {
        var network = BuildNetwork();
        var analytical = AnalyticalWeightGradients(network, layerIndex);

        for (var weightIndex = 0; weightIndex < analytical.Count; weightIndex++)
        {
            NumericalWeightGradient(layerIndex, weightIndex)
                .Should()
                .BeApproximately(
                    analytical[weightIndex],
                    Tolerance,
                    $"weight {weightIndex} of layer {layerIndex} should match its finite difference estimate");
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void BiasGradients_MatchNumericalEstimates(int layerIndex)
    {
        var network = BuildNetwork();
        var analytical = AnalyticalBiasGradients(network, layerIndex);

        for (var biasIndex = 0; biasIndex < analytical.Count; biasIndex++)
        {
            NumericalBiasGradient(layerIndex, biasIndex)
                .Should()
                .BeApproximately(
                    analytical[biasIndex],
                    Tolerance,
                    $"bias {biasIndex} of layer {layerIndex} should match its finite difference estimate");
        }
    }

    private static FeedForwardNetwork BuildNetwork() =>
        NetworkInitializer.Create(NetworkTopology.Create([3, 4, 2]), RandomSeed.Create(7));

    /// <summary>
    /// Recovers the gradient the trainer applied, by measuring how far one descent step at a
    /// known learning rate moved each parameter.
    /// </summary>
    private static IReadOnlyList<double> AnalyticalWeightGradients(FeedForwardNetwork network, int layerIndex)
    {
        var (before, after) = DescendOnce(network);

        return [.. before.Layers[layerIndex].Weights.Zip(
            after.Layers[layerIndex].Weights,
            static (start, end) => start - end)];
    }

    private static IReadOnlyList<double> AnalyticalBiasGradients(FeedForwardNetwork network, int layerIndex)
    {
        var (before, after) = DescendOnce(network);

        return [.. before.Layers[layerIndex].Biases.Zip(
            after.Layers[layerIndex].Biases,
            static (start, end) => start - end)];
    }

    private static (NetworkSnapshot Before, NetworkSnapshot After) DescendOnce(FeedForwardNetwork network)
    {
        // A learning rate of exactly one makes the parameter delta equal to the gradient.
        var outcome = network.TrainOnBatch(Inputs, Expected, LearningRate.Create(1d));

        return (NetworkSerializer.ToSnapshot(network), NetworkSerializer.ToSnapshot(outcome.Network));
    }

    private static double NumericalWeightGradient(int layerIndex, int weightIndex) => CentralDifference(
        offset => LossWith(layerIndex, weights => Adjust(weights, weightIndex, offset), biases => biases));

    private static double NumericalBiasGradient(int layerIndex, int biasIndex) => CentralDifference(
        offset => LossWith(layerIndex, weights => weights, biases => Adjust(biases, biasIndex, offset)));

    private static double CentralDifference(Func<double, double> lossAtOffset) =>
        (lossAtOffset(Perturbation) - lossAtOffset(-Perturbation)) / (2d * Perturbation);

    private static double LossWith(
        int layerIndex,
        Func<IReadOnlyList<double>, IReadOnlyList<double>> adjustWeights,
        Func<IReadOnlyList<double>, IReadOnlyList<double>> adjustBiases)
    {
        var snapshot = NetworkSerializer.ToSnapshot(BuildNetwork());
        var layers = snapshot.Layers.ToArray();
        var target = layers[layerIndex];

        layers[layerIndex] = target with
        {
            Weights = adjustWeights(target.Weights),
            Biases = adjustBiases(target.Biases),
        };

        return NetworkSerializer.Default
            .FromSnapshot(new NetworkSnapshot(layers, snapshot.LossFunction))
            .Evaluate(Inputs, Expected)
            .MeanLoss;
    }

    private static double[] Adjust(IReadOnlyList<double> values, int index, double offset)
    {
        var adjusted = values.ToArray();
        adjusted[index] += offset;

        return adjusted;
    }
}
