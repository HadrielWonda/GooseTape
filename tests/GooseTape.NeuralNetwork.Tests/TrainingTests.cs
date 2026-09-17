using GooseTape.NeuralNetwork.Data;
using GooseTape.NeuralNetwork.Training;

namespace GooseTape.NeuralNetwork.Tests;

public sealed class TrainingTests
{
    private const int ExamplesPerClass = 40;
    private const int DatasetSeed = 11;

    [Fact]
    public void Training_OverSeveralEpochs_DrivesAccuracyUp()
    {
        var dataset = SyntheticDatasetSource.Create(ExamplesPerClass, DatasetSeed).Build();
        var trainer = EpochTrainer.Create(TrainingSchedule.Create(0.5d, 16), RandomSeed.Create(3));
        var network = NetworkInitializer.Create(
            NetworkTopology.Create([dataset.PixelCount, 32, DigitLabel.ClassCount]),
            RandomSeed.Create(3));

        var startingAccuracy = network.Evaluate(
            dataset.AsSingleBatch().Inputs,
            dataset.AsSingleBatch().Expected).Accuracy;

        var epoch = EpochNumber.Create(1);

        for (var pass = 0; pass < 25; pass++)
        {
            network = trainer.Run(network, dataset, epoch).Network;
            epoch = epoch.Next();
        }

        var batch = dataset.AsSingleBatch();
        var finalMetrics = network.Evaluate(batch.Inputs, batch.Expected);

        startingAccuracy.Should().BeLessThan(0.5d, "an untrained network should be near chance");
        finalMetrics.Accuracy.Should().BeGreaterThan(0.95d, "the synthetic classes are linearly separable");
        finalMetrics.MeanLoss.Should().BeLessThan(0.3d);
    }

    [Fact]
    public void Training_ReducesLossMonotonicallyOverTheFirstEpochs()
    {
        var dataset = SyntheticDatasetSource.Create(ExamplesPerClass, DatasetSeed).Build();
        var trainer = EpochTrainer.Create(TrainingSchedule.Create(0.3d, 16), RandomSeed.Create(5));
        var network = NetworkInitializer.Create(
            NetworkTopology.Create([dataset.PixelCount, 24, DigitLabel.ClassCount]),
            RandomSeed.Create(5));

        var losses = new List<double>();
        var epoch = EpochNumber.Create(1);

        for (var pass = 0; pass < 6; pass++)
        {
            var outcome = trainer.Run(network, dataset, epoch);
            losses.Add(outcome.Metrics.MeanLoss);
            network = outcome.Network;
            epoch = epoch.Next();
        }

        losses.Should().BeInDescendingOrder();
    }

    [Fact]
    public void Training_LeavesTheOriginalNetworkUntouched()
    {
        // Immutability is what makes an epoch safe to retry: a failed epoch must not have
        // modified the network the orchestrator would resume from.
        var dataset = SyntheticDatasetSource.Create(4, DatasetSeed).Build();
        var batch = dataset.AsSingleBatch();
        var network = NetworkInitializer.Create(
            NetworkTopology.Create([dataset.PixelCount, 8, DigitLabel.ClassCount]),
            RandomSeed.Create(1));

        var before = network.Evaluate(batch.Inputs, batch.Expected);
        _ = network.TrainOnBatch(batch.Inputs, batch.Expected, LearningRate.Create(0.5d));
        var after = network.Evaluate(batch.Inputs, batch.Expected);

        after.Should().Be(before);
    }

    [Fact]
    public void Initialisation_WithTheSameSeed_IsReproducible()
    {
        var topology = NetworkTopology.Create([6, 5, 4]);

        var first = NetworkInitializer.Create(topology, RandomSeed.Create(42));
        var second = NetworkInitializer.Create(topology, RandomSeed.Create(42));

        first.Layers[0].Parameters.Weights.ToArray()
            .Should().Equal(second.Layers[0].Parameters.Weights.ToArray());
    }

    [Fact]
    public void Initialisation_WithDifferentSeeds_Diverges()
    {
        var topology = NetworkTopology.Create([6, 5, 4]);

        var first = NetworkInitializer.Create(topology, RandomSeed.Create(1));
        var second = NetworkInitializer.Create(topology, RandomSeed.Create(2));

        first.Layers[0].Parameters.Weights.ToArray()
            .Should().NotEqual(second.Layers[0].Parameters.Weights.ToArray());
    }

    [Fact]
    public void Epoch_ReRunFromTheSameNetwork_ProducesIdenticalResults()
    {
        // Replay depends on this: the same epoch number must shuffle the data the same way,
        // in this process or any other.
        var dataset = SyntheticDatasetSource.Create(8, DatasetSeed).Build();
        var trainer = EpochTrainer.Create(TrainingSchedule.Create(0.4d, 8), RandomSeed.Create(9));
        var network = NetworkInitializer.Create(
            NetworkTopology.Create([dataset.PixelCount, 12, DigitLabel.ClassCount]),
            RandomSeed.Create(9));

        var first = trainer.Run(network, dataset, EpochNumber.Create(4));
        var second = trainer.Run(network, dataset, EpochNumber.Create(4));

        second.Metrics.Should().Be(first.Metrics);
        second.Network.Layers[0].Parameters.Weights.ToArray()
            .Should().Equal(first.Network.Layers[0].Parameters.Weights.ToArray());
    }

    [Fact]
    public void Epoch_WithDifferentEpochNumbers_ShufflesDifferently()
    {
        var dataset = SyntheticDatasetSource.Create(8, DatasetSeed).Build();
        var trainer = EpochTrainer.Create(TrainingSchedule.Create(0.4d, 8), RandomSeed.Create(9));
        var network = NetworkInitializer.Create(
            NetworkTopology.Create([dataset.PixelCount, 12, DigitLabel.ClassCount]),
            RandomSeed.Create(9));

        var first = trainer.Run(network, dataset, EpochNumber.Create(1));
        var second = trainer.Run(network, dataset, EpochNumber.Create(2));

        second.Network.Layers[0].Parameters.Weights.ToArray()
            .Should().NotEqual(first.Network.Layers[0].Parameters.Weights.ToArray());
    }
}
