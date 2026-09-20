using System.Text;
using GooseTape.NeuralNetwork.Checkpoints;
using GooseTape.NeuralNetwork.Data;
using GooseTape.NeuralNetwork.Training;

namespace GooseTape.NeuralNetwork.Tests;

/// <summary>
/// Covers what the checkpoint store does when a process dies mid-write, when two writers race,
/// and when a stored file is damaged.
/// </summary>
/// <remarks>
/// Atomic writes are only half of the durability claim. The property that matters is that a
/// checkpoint is either entirely present and loadable, or not there at all — never a partial
/// file that loads into a plausible but wrong network.
/// </remarks>
public sealed class CrashConsistencyTests : IDisposable
{
    private readonly string _directory =
        Path.Combine(Path.GetTempPath(), $"goosetape-crash-tests-{Guid.NewGuid():N}");

    [Fact]
    public async Task ACrashPartWayThroughAWrite_LeavesThePreviousCheckpointIntact()
    {
        // A crashed write leaves its temporary file behind, never a half-written checkpoint,
        // because the store only publishes by moving the completed file into place.
        var store = FileSystemCheckpointStore.Create(_directory);
        var identifier = CheckpointId.Create("run-a-epoch-1");
        await store.SaveAsync(identifier, SnapshotOf(TrainedNetwork(seed: 1)));
        var before = await store.LoadAsync(identifier);

        await File.WriteAllTextAsync(
            Path.Combine(_directory, $"{identifier.Value}.checkpoint.json.{Guid.NewGuid():N}.tmp"),
            "{ \"Layers\": [ truncated");

        var after = await store.LoadAsync(identifier);

        after.Layers[0].Weights.Should().Equal(before.Layers[0].Weights);
        after.Metadata.Should().BeEquivalentTo(before.Metadata);
    }

    [Fact]
    public async Task ATruncatedCheckpoint_IsRejectedRatherThanPartiallyLoaded()
    {
        var store = FileSystemCheckpointStore.Create(_directory);
        var identifier = CheckpointId.Create("run-a-epoch-2");
        await store.SaveAsync(identifier, SnapshotOf(TrainedNetwork(seed: 2)));

        var path = Path.Combine(_directory, $"{identifier.Value}.checkpoint.json");
        var contents = await File.ReadAllTextAsync(path);
        await File.WriteAllTextAsync(path, contents[..(contents.Length / 2)]);

        var act = async () => await store.LoadAsync(identifier);

        // Half a JSON document is not a checkpoint, and must not load as one.
        await act.Should().ThrowAsync<System.Text.Json.JsonException>();
    }

    [Fact]
    public async Task ACheckpointOfGarbage_IsRejected()
    {
        var store = FileSystemCheckpointStore.Create(_directory);
        var identifier = CheckpointId.Create("run-a-garbage");
        await File.WriteAllBytesAsync(
            Path.Combine(_directory, $"{identifier.Value}.checkpoint.json"),
            Encoding.UTF8.GetBytes("not json at all"));

        var act = async () => await store.LoadAsync(identifier);

        await act.Should().ThrowAsync<System.Text.Json.JsonException>();
    }

    [Fact]
    public async Task ConcurrentWritersOfTheSameCheckpoint_LeaveOneCompleteFile()
    {
        // Two workers retrying the same epoch write identical content. Whichever lands last, the
        // reader must still see one whole checkpoint rather than a mixture of the two.
        var store = FileSystemCheckpointStore.Create(_directory);
        var identifier = CheckpointId.Create("run-a-contended");
        var snapshot = SnapshotOf(TrainedNetwork(seed: 3));

        await Task.WhenAll(Enumerable.Range(0, 8).Select(_ => store.SaveAsync(identifier, snapshot)));

        var loaded = await store.LoadAsync(identifier);

        loaded.Layers.Should().HaveCount(snapshot.Layers.Count);
        loaded.Layers[0].Weights.Should().Equal(snapshot.Layers[0].Weights);
        Directory.GetFiles(_directory, "*.tmp").Should().BeEmpty("finished writes should leave no temporary files");
    }

    [Fact]
    public async Task AnInterruptedRunResumesFromItsLastGoodCheckpoint()
    {
        // The scenario the durable feature relies on: epochs 1 and 2 are safe on disk, epoch 3
        // died mid-flight, and training continues from epoch 2 without repeating the earlier work.
        var store = FileSystemCheckpointStore.Create(_directory);
        var dataset = SyntheticDatasetSource.Create(6, 17).Build();
        var trainer = EpochTrainer.Create(TrainingSchedule.Create(0.4d, 8), RandomSeed.Create(5));
        var network = NetworkInitializer.Create(
            NetworkTopology.Create([dataset.PixelCount, 12, DigitLabel.ClassCount]),
            RandomSeed.Create(5));

        for (var epoch = 1; epoch <= 2; epoch++)
        {
            network = trainer.Run(network, dataset, EpochNumber.Create(epoch)).Network;
            await store.SaveAsync(CheckpointId.ForEpoch("interrupted", epoch), NetworkSerializer.ToSnapshot(network));
        }

        var uninterrupted = trainer.Run(network, dataset, EpochNumber.Create(3)).Network;

        // Now "resume": reload epoch 2 from disk, as a fresh process would, and run epoch 3.
        var restored = NetworkSerializer.Default.FromSnapshot(
            await store.LoadAsync(CheckpointId.ForEpoch("interrupted", 2)));
        var resumed = trainer.Run(restored, dataset, EpochNumber.Create(3)).Network;

        var batch = dataset.AsSingleBatch();
        resumed.Predict(batch.Inputs).ToArray()
            .Should().Equal(uninterrupted.Predict(batch.Inputs).ToArray(),
                "resuming from a checkpoint must produce the same network as never having stopped");
        (await store.ExistsAsync(CheckpointId.ForEpoch("interrupted", 3))).Should().BeFalse();
    }

    private static FeedForwardNetwork TrainedNetwork(int seed)
    {
        var dataset = SyntheticDatasetSource.Create(3, seed).Build();
        var trainer = EpochTrainer.Create(TrainingSchedule.Create(0.3d, 8), RandomSeed.Create(seed));
        var network = NetworkInitializer.Create(
            NetworkTopology.Create([dataset.PixelCount, 8, DigitLabel.ClassCount]),
            RandomSeed.Create(seed));

        return trainer.Run(network, dataset, EpochNumber.Create(1)).Network;
    }

    private static NetworkSnapshot SnapshotOf(FeedForwardNetwork network) => NetworkSerializer.ToSnapshot(
        network,
        new CheckpointMetadata(
            "run-a",
            Epoch: 1,
            ParentCheckpointId: "run-a-epoch-0",
            DatasetId: "synthetic:3:1",
            Topology: "20-8-10",
            Seed: 1,
            LearningRate: 0.3d,
            BatchSize: 8,
            CreatedAt: DateTimeOffset.UnixEpoch,
            FormatVersion: CheckpointMetadata.CurrentFormatVersion));

    public void Dispose()
    {
        if (Directory.Exists(_directory))
        {
            Directory.Delete(_directory, recursive: true);
        }
    }
}
