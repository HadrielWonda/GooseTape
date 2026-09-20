using BenchmarkDotNet.Attributes;
using GooseTape.NeuralNetwork.Checkpoints;
using GooseTape.NeuralNetwork.Data;
using GooseTape.NeuralNetwork.Training;

namespace GooseTape.Benchmarks;

/// <summary>
/// Measures a full training step and a checkpoint round trip.
/// </summary>
/// <remarks>
/// Training returns a new network for every batch, which is what makes an epoch safe to retry.
/// These benchmarks exist to price that decision: the allocation figures show what immutability
/// costs per update, and per epoch at MNIST scale.
/// </remarks>
[MemoryDiagnoser]
public class TrainingBenchmarks
{
    private const int PixelCount = 784;
    private const int HiddenNeurons = 128;

    private FeedForwardNetwork _network = default!;
    private TrainingBatch _batch = default!;
    private ImageDataset _epochSample = default!;
    private NetworkSnapshot _snapshot = default!;
    private string _serialised = default!;

    /// <summary>Gets or sets the number of examples per batch.</summary>
    [Params(32, 64, 128)]
    public int BatchSize { get; set; }

    /// <summary>Builds a network, a batch, and a small epoch sample once per parameter set.</summary>
    [GlobalSetup]
    public void Setup()
    {
        _network = NetworkInitializer.Create(
            NetworkTopology.Create([PixelCount, HiddenNeurons, DigitLabel.ClassCount]),
            RandomSeed.Create(20_260_920));

        // Images the width of an MNIST digit, generated so the benchmark needs no download.
        var images = Enumerable.Range(0, 512)
            .Select(index => LabelledImage.Create(
                PixelGrid.FromGreyscale(PixelsFor(index)),
                DigitLabel.Create(index % DigitLabel.ClassCount)))
            .ToList();

        _epochSample = ImageDataset.From(images);
        _batch = _epochSample.TakeAtMost(BatchSize).AsSingleBatch();
        _snapshot = NetworkSerializer.ToSnapshot(_network);
        _serialised = System.Text.Json.JsonSerializer.Serialize(_snapshot);
    }

    /// <summary>One parameter update: forward, backward, and a new network.</summary>
    [Benchmark(Baseline = true)]
    public FeedForwardNetwork TrainOnOneBatch() =>
        _network.TrainOnBatch(_batch.Inputs, _batch.Expected, LearningRate.Create(0.1d)).Network;

    /// <summary>A forward pass only, for comparison against a full update.</summary>
    [Benchmark]
    public TrainingMetrics EvaluateOneBatch() => _network.Evaluate(_batch.Inputs, _batch.Expected);

    /// <summary>An epoch over 512 examples, including the shuffle.</summary>
    [Benchmark]
    public FeedForwardNetwork TrainOneEpochOverASample() =>
        EpochTrainer.Create(TrainingSchedule.Create(0.1d, BatchSize), RandomSeed.Create(1))
            .Run(_network, _epochSample, EpochNumber.Create(1))
            .Network;

    /// <summary>Flattening a network for storage.</summary>
    [Benchmark]
    public string SerialiseCheckpoint() => System.Text.Json.JsonSerializer.Serialize(_snapshot);

    /// <summary>Rebuilding a network from storage, as a resume would.</summary>
    [Benchmark]
    public FeedForwardNetwork DeserialiseCheckpoint() => NetworkSerializer.Default.FromSnapshot(
        System.Text.Json.JsonSerializer.Deserialize<NetworkSnapshot>(_serialised)!);

    private static byte[] PixelsFor(int index)
    {
        var pixels = new byte[PixelCount];
        var random = new Random(index);
        random.NextBytes(pixels);

        return pixels;
    }
}
