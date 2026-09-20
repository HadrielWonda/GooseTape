using System.Text.Json;
using GooseTape.Functions.Host.Ductape;
using GooseTape.Functions.Host.Training;
using GooseTape.NeuralNetwork.Data;
using GooseTape.NeuralNetwork.Maths;

namespace GooseTape.Functions.Host.Operations;

/// <summary>
/// Classifies a single image with a checkpointed network.
/// </summary>
public sealed class ClassifyDigitOperation : IPortableFunctionOperation
{
    /// <summary>The operation name declared in the portable function contract.</summary>
    public const string OperationName = "classify-digit";

    private const int MostPixels = 65_536;

    private readonly NetworkCheckpoints _checkpoints;
    private readonly DatasetProvider _datasets;

    /// <summary>
    /// Creates the operation.
    /// </summary>
    /// <param name="checkpoints">Loads the network to classify with.</param>
    /// <param name="datasets">Supplies the expected input width.</param>
    /// <exception cref="ArgumentNullException">Thrown when either argument is null.</exception>
    public ClassifyDigitOperation(NetworkCheckpoints checkpoints, DatasetProvider datasets)
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
        var pixels = reader.RequireNumberArray("pixels", 1, MostPixels);

        var dataset = await _datasets.TrainingAsync().ConfigureAwait(false);
        GuardPixelCount(pixels.Count, dataset.PixelCount);

        var restored = await _checkpoints.LoadAsync(checkpointId, cancellationToken).ConfigureAwait(false);
        var network = restored.Network;
        var distribution = network.Predict(Matrix.FromRow(NormaliseOrReject(pixels)));
        var digit = distribution.IndexOfLargestInRow(0);

        return new ClassifyDigitOutput(digit, distribution[0, digit], distribution.ToArray());
    }

    private static void GuardPixelCount(int supplied, int expected)
    {
        if (supplied == expected)
        {
            return;
        }

        throw new PortableFunctionException(
            "FUNCTION_INPUT_INVALID",
            $"Function input field pixels must hold exactly {expected} values to match the dataset, but held {supplied}.");
    }

    private static double[] NormaliseOrReject(IReadOnlyList<double> pixels)
    {
        try
        {
            return PixelGrid.FromNormalised(pixels).Intensities.ToArray();
        }
        catch (ArgumentException failure)
        {
            throw new PortableFunctionException("FUNCTION_INPUT_INVALID", failure.Message);
        }
    }

    /// <summary>
    /// What classification reports back to the calling feature.
    /// </summary>
    /// <param name="Digit">The most likely digit.</param>
    /// <param name="Confidence">The probability assigned to that digit.</param>
    /// <param name="Distribution">The full probability distribution across all ten digits.</param>
    private sealed record ClassifyDigitOutput(int Digit, double Confidence, IReadOnlyList<double> Distribution);
}
