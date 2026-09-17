using System.Diagnostics;
using System.Text.Json;
using GooseTape.Functions.Host.Ductape;
using GooseTape.Functions.Host.Training;
using GooseTape.NeuralNetwork.Training;

namespace GooseTape.Functions.Host.Operations;

/// <summary>
/// Runs exactly one training epoch and stores the resulting network as a new checkpoint.
/// </summary>
/// <remarks>
/// <para>
/// This is the unit of work the durable training feature checkpoints around. It reads one
/// checkpoint, writes another, and holds no state between calls, so the orchestrator can retry,
/// resume or replay a single epoch without the rest of the run being affected.
/// </para>
/// <para>
/// The operation is safe to declare idempotent. Given the same input checkpoint, epoch number
/// and seed, the shuffle order and every parameter update are fully determined, so a retry
/// recomputes an identical result and overwrites the output checkpoint with the same bytes.
/// </para>
/// </remarks>
public sealed class TrainEpochOperation : IPortableFunctionOperation
{
    /// <summary>The operation name declared in the portable function contract.</summary>
    public const string OperationName = "train-epoch";

    private const int MostEpochs = 10_000;
    private const int LargestBatch = 8_192;

    private readonly NetworkCheckpoints _checkpoints;
    private readonly DatasetProvider _datasets;

    /// <summary>
    /// Creates the operation.
    /// </summary>
    /// <param name="checkpoints">Loads the starting network and stores the trained one.</param>
    /// <param name="datasets">Supplies the training data.</param>
    /// <exception cref="ArgumentNullException">Thrown when either argument is null.</exception>
    public TrainEpochOperation(NetworkCheckpoints checkpoints, DatasetProvider datasets)
    {
        ArgumentNullException.ThrowIfNull(checkpoints);
        ArgumentNullException.ThrowIfNull(datasets);

        _checkpoints = checkpoints;
        _datasets = datasets;
    }

    /// <inheritdoc />
    public string Name => OperationName;

    /// <inheritdoc />
    public async Task<object> ExecuteAsync(
        JsonElement input,
        InvocationContext context,
        CancellationToken cancellationToken)
    {
        var request = ReadRequest(InvocationInput.From(input));
        var startedAt = Stopwatch.GetTimestamp();

        var dataset = await _datasets.TrainingAsync().ConfigureAwait(false);
        var sample = request.SampleSize is null ? dataset : dataset.TakeAtMost(request.SampleSize.Value);

        var network = await _checkpoints
            .LoadAsync(NetworkCheckpoints.Parse(request.FromCheckpoint), cancellationToken)
            .ConfigureAwait(false);

        var outcome = Train(request, network, sample);
        var checkpointId = NetworkCheckpoints.Parse($"{request.RunIdentifier}-epoch-{request.Epoch}");

        await _checkpoints.SaveAsync(checkpointId, outcome.Network, cancellationToken).ConfigureAwait(false);

        return new TrainEpochOutput(
            checkpointId.Value,
            request.Epoch,
            outcome.Metrics.MeanLoss,
            outcome.Metrics.Accuracy,
            outcome.Metrics.ExampleCount,
            Stopwatch.GetElapsedTime(startedAt).TotalMilliseconds);
    }

    private static EpochOutcome Train(EpochRequest request, FeedForwardNetwork network, NeuralNetwork.Data.ImageDataset sample)
    {
        try
        {
            var trainer = EpochTrainer.Create(
                TrainingSchedule.Create(request.LearningRate, request.BatchSize),
                RandomSeed.Create(request.Seed));

            return trainer.Run(network, sample, EpochNumber.Create(request.Epoch));
        }
        catch (ArgumentException failure)
        {
            throw new PortableFunctionException("FUNCTION_INPUT_INVALID", failure.Message);
        }
    }

    private static EpochRequest ReadRequest(InvocationInput reader) => new(
        reader.RequireString("run_id"),
        reader.RequireInteger("epoch", 1, MostEpochs),
        reader.RequireString("from_checkpoint"),
        reader.RequireNumber("learning_rate", 1e-8d, 1d),
        reader.RequireInteger("batch_size", 1, LargestBatch),
        reader.RequireInteger("seed", int.MinValue, int.MaxValue),
        reader.OptionalInteger("train_sample_size", 1, int.MaxValue));

    private sealed record EpochRequest(
        string RunIdentifier,
        int Epoch,
        string FromCheckpoint,
        double LearningRate,
        int BatchSize,
        int Seed,
        int? SampleSize);

    /// <summary>
    /// What one epoch reports back to the calling feature.
    /// </summary>
    /// <param name="CheckpointId">The checkpoint written at the end of the epoch.</param>
    /// <param name="Epoch">The epoch that was run.</param>
    /// <param name="MeanLoss">The example-weighted mean training loss across the epoch.</param>
    /// <param name="Accuracy">The example-weighted training accuracy across the epoch.</param>
    /// <param name="ExampleCount">How many examples were seen.</param>
    /// <param name="DurationMs">How long the epoch took, in milliseconds.</param>
    private sealed record TrainEpochOutput(
        string CheckpointId,
        int Epoch,
        double MeanLoss,
        double Accuracy,
        int ExampleCount,
        double DurationMs);
}
