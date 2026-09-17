using System.Text.Json;
using GooseTape.Functions.Host.Ductape;
using GooseTape.Functions.Host.Training;
using GooseTape.NeuralNetwork.Checkpoints;
using GooseTape.NeuralNetwork.Training;

namespace GooseTape.Functions.Host.Operations;

/// <summary>
/// Builds an untrained network for a run and stores it as the epoch zero checkpoint.
/// </summary>
/// <remarks>
/// Kept separate from training so a durable run has a single, reproducible starting point that
/// a replay can return to. Given the same run identifier, topology and seed, this always
/// produces byte-identical weights.
/// </remarks>
public sealed class InitializeNetworkOperation : IPortableFunctionOperation
{
    /// <summary>The operation name declared in the portable function contract.</summary>
    public const string OperationName = "initialize-network";

    private const int FewestLayerSizes = 3;
    private const int MostLayerSizes = 8;

    private readonly NetworkCheckpoints _checkpoints;
    private readonly DatasetProvider _datasets;

    /// <summary>
    /// Creates the operation.
    /// </summary>
    /// <param name="checkpoints">Stores the resulting network.</param>
    /// <param name="datasets">Supplies the training data the topology is checked against.</param>
    /// <exception cref="ArgumentNullException">Thrown when either argument is null.</exception>
    public InitializeNetworkOperation(NetworkCheckpoints checkpoints, DatasetProvider datasets)
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
        var runIdentifier = reader.RequireString("run_id");
        var sizes = reader.RequireIntegerArray("topology", FewestLayerSizes, MostLayerSizes);
        var seed = reader.RequireInteger("seed", int.MinValue, int.MaxValue);

        var dataset = await _datasets.TrainingAsync().ConfigureAwait(false);
        GuardTopologyMatchesData(sizes, dataset.PixelCount);

        var topology = BuildTopology(sizes);
        var network = NetworkInitializer.Create(topology, RandomSeed.Create(seed));
        var checkpointId = NetworkCheckpoints.Parse($"{runIdentifier}-epoch-0");

        await _checkpoints.SaveAsync(checkpointId, network, cancellationToken).ConfigureAwait(false);

        return new InitializeNetworkOutput(
            checkpointId.Value,
            topology.DenseLayerCount,
            CountParameters(sizes),
            topology.InputSize.Value,
            topology.OutputSize.Value);
    }

    private static NetworkTopology BuildTopology(IReadOnlyList<int> sizes)
    {
        try
        {
            return NetworkTopology.Create(sizes);
        }
        catch (Exception failure) when (failure is ArgumentException or ArgumentOutOfRangeException)
        {
            throw new PortableFunctionException("FUNCTION_INPUT_INVALID", failure.Message);
        }
    }

    private static void GuardTopologyMatchesData(IReadOnlyList<int> sizes, int pixelCount)
    {
        if (sizes[0] == pixelCount)
        {
            return;
        }

        throw new PortableFunctionException(
            "TOPOLOGY_MISMATCH",
            $"The first topology entry is the input width and must equal the dataset pixel count of {pixelCount}, but was {sizes[0]}.");
    }

    private static int CountParameters(IReadOnlyList<int> sizes)
    {
        var total = 0;

        for (var index = 0; index < sizes.Count - 1; index++)
        {
            total += (sizes[index] * sizes[index + 1]) + sizes[index + 1];
        }

        return total;
    }

    /// <summary>
    /// What initialisation reports back to the calling feature.
    /// </summary>
    /// <param name="CheckpointId">The epoch zero checkpoint the run starts from.</param>
    /// <param name="LayerCount">The number of dense layers created.</param>
    /// <param name="ParameterCount">The total number of weights and biases.</param>
    /// <param name="InputSize">The number of inputs the network accepts.</param>
    /// <param name="OutputSize">The number of classes the network predicts.</param>
    private sealed record InitializeNetworkOutput(
        string CheckpointId,
        int LayerCount,
        int ParameterCount,
        int InputSize,
        int OutputSize);
}
