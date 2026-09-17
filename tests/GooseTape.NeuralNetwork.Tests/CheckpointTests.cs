using GooseTape.NeuralNetwork.Checkpoints;
using GooseTape.NeuralNetwork.Data;
using GooseTape.NeuralNetwork.Training;

namespace GooseTape.NeuralNetwork.Tests;

public sealed class CheckpointTests : IDisposable
{
    private readonly string _directory =
        Path.Combine(Path.GetTempPath(), $"goosetape-tests-{Guid.NewGuid():N}");

    [Fact]
    public async Task SaveThenLoad_ReproducesIdenticalPredictions()
    {
        var dataset = SyntheticDatasetSource.Create(4, 21).Build();
        var batch = dataset.AsSingleBatch();
        var store = FileSystemCheckpointStore.Create(_directory);
        var trained = TrainBriefly(dataset);

        await store.SaveAsync(CheckpointId.Create("run-a-epoch-1"), NetworkSerializer.ToSnapshot(trained));
        var restored = NetworkSerializer.Default.FromSnapshot(
            await store.LoadAsync(CheckpointId.Create("run-a-epoch-1")));

        restored.Predict(batch.Inputs).ToArray()
            .Should().Equal(trained.Predict(batch.Inputs).ToArray());
    }

    [Fact]
    public async Task Load_ForAnUnknownCheckpoint_ThrowsNamingIt()
    {
        var store = FileSystemCheckpointStore.Create(_directory);

        var act = async () => await store.LoadAsync(CheckpointId.Create("never-written"));

        await act.Should().ThrowAsync<CheckpointNotFoundException>().WithMessage("*never-written*");
    }

    [Fact]
    public async Task Exists_ReflectsWhetherASnapshotWasWritten()
    {
        var store = FileSystemCheckpointStore.Create(_directory);
        var identifier = CheckpointId.Create("presence-probe");

        (await store.ExistsAsync(identifier)).Should().BeFalse();
        await store.SaveAsync(identifier, NetworkSerializer.ToSnapshot(TrainBriefly(SyntheticDatasetSource.Create(2, 3).Build())));
        (await store.ExistsAsync(identifier)).Should().BeTrue();
    }

    [Fact]
    public async Task Save_OverAnExistingCheckpoint_ReplacesIt()
    {
        var store = FileSystemCheckpointStore.Create(_directory);
        var identifier = CheckpointId.Create("overwritten");
        var dataset = SyntheticDatasetSource.Create(4, 7).Build();

        await store.SaveAsync(identifier, NetworkSerializer.ToSnapshot(TrainBriefly(dataset)));
        var second = TrainBriefly(dataset, epochs: 5);
        await store.SaveAsync(identifier, NetworkSerializer.ToSnapshot(second));

        var restored = NetworkSerializer.Default.FromSnapshot(await store.LoadAsync(identifier));
        var batch = dataset.AsSingleBatch();

        restored.Predict(batch.Inputs).ToArray().Should().Equal(second.Predict(batch.Inputs).ToArray());
    }

    [Theory]
    [InlineData("../escape")]
    [InlineData("..\escape")]
    [InlineData("nested/path")]
    [InlineData("with space")]
    [InlineData("")]
    [InlineData("   ")]
    public void CheckpointId_RejectsAnythingThatCouldEscapeTheStoreDirectory(string candidate)
    {
        var act = () => CheckpointId.Create(candidate);

        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void CheckpointId_RejectsAnOverLongIdentifier()
    {
        var act = () => CheckpointId.Create(new string('a', 129));

        act.Should().Throw<ArgumentException>().WithMessage("*at most 128 characters*");
    }

    [Fact]
    public void Snapshot_WithAWeightCountThatContradictsItsShape_IsRejected()
    {
        var snapshot = new NetworkSnapshot(
            [new LayerSnapshot(3, 2, "relu", [1, 2, 3], [0, 0]), new LayerSnapshot(2, 2, "softmax", [1, 2, 3, 4], [0, 0])],
            "cross-entropy");

        var act = () => NetworkSerializer.Default.FromSnapshot(snapshot);

        act.Should().Throw<InvalidDataException>().WithMessage("*needs 6 weights*holds 3*");
    }

    [Fact]
    public void Snapshot_NamingAnUnknownActivation_IsRejected()
    {
        var snapshot = new NetworkSnapshot(
            [new LayerSnapshot(2, 2, "mystery", [1, 2, 3, 4], [0, 0]), new LayerSnapshot(2, 2, "softmax", [1, 2, 3, 4], [0, 0])],
            "cross-entropy");

        var act = () => NetworkSerializer.Default.FromSnapshot(snapshot);

        act.Should().Throw<NotSupportedException>().WithMessage("*mystery*");
    }

    private static FeedForwardNetwork TrainBriefly(ImageDataset dataset, int epochs = 2)
    {
        var trainer = EpochTrainer.Create(TrainingSchedule.Create(0.3d, 8), RandomSeed.Create(13));
        var network = NetworkInitializer.Create(
            NetworkTopology.Create([dataset.PixelCount, 10, DigitLabel.ClassCount]),
            RandomSeed.Create(13));

        var epoch = EpochNumber.Create(1);

        for (var pass = 0; pass < epochs; pass++)
        {
            network = trainer.Run(network, dataset, epoch).Network;
            epoch = epoch.Next();
        }

        return network;
    }

    public void Dispose()
    {
        if (Directory.Exists(_directory))
        {
            Directory.Delete(_directory, recursive: true);
        }
    }
}
