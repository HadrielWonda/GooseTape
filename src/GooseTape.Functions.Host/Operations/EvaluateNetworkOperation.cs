using System.Text.Json;
using GooseTape.Functions.Host.Ductape;
using GooseTape.Functions.Host.Training;

namespace GooseTape.Functions.Host.Operations;

/// <summary>
/// Measures a checkpointed network against held out test data.
/// </summary>
/// <remarks>
/// Deliberately reads the test dataset rather than the training dataset. The accuracy a
/// training epoch reports is measured on data the network is being fitted to, so it drifts
/// optimistic; this is the number a run should actually be judged on.
/// </remarks>
public sealed class EvaluateNetworkOperation : IPortableFunctionOperation
{
    /// <summary>The operation name declared in the portable function contract.</summary>
    public const string OperationName = "evaluate-network";

    private readonly NetworkCheckpoints _checkpoints;
    private readonly DatasetProvider _datasets;

    /// <summary>
    /// Creates the operation.
    /// </summary>
    /// <param name="checkpoints">Loads the network to evaluate.</param>
    /// <param name="datasets">Supplies the held out test data.</param>
    /// <exception cref="ArgumentNullException">Thrown when either argument is null.</exception>
    public EvaluateNetworkOperation(NetworkCheckpoints checkpoints, DatasetProvider datasets)
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
        var reader = InvocationInput.From(input);
        var checkpointId = NetworkCheckpoints.Parse(reader.RequireString("checkpoint_id"));
        var sampleSize = reader.OptionalInteger("sample_size", 1, int.MaxValue);

        var dataset = await _datasets.TestAsync().ConfigureAwait(false);
        var sample = sampleSize is null ? dataset : dataset.TakeAtMost(sampleSize.Value);
        var restored = await _checkpoints.LoadAsync(checkpointId, cancellationToken).ConfigureAwait(false);
        var network = restored.Network;

        var batch = sample.AsSingleBatch();
        var metrics = network.Evaluate(batch.Inputs, batch.Expected);

        return new EvaluateNetworkOutput(
            checkpointId.Value,
            metrics.Accuracy,
            metrics.MeanLoss,
            metrics.ExampleCount);
    }

    /// <summary>
    /// What evaluation reports back to the calling feature.
    /// </summary>
    /// <param name="CheckpointId">The checkpoint that was measured.</param>
    /// <param name="Accuracy">The fraction of test examples classified correctly.</param>
    /// <param name="MeanLoss">The mean loss across the test examples.</param>
    /// <param name="ExampleCount">How many test examples were measured.</param>
    private sealed record EvaluateNetworkOutput(
        string CheckpointId,
        double Accuracy,
        double MeanLoss,
        int ExampleCount);
}
