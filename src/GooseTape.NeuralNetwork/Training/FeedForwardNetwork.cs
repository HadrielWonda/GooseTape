using GooseTape.NeuralNetwork.Layers;
using GooseTape.NeuralNetwork.Losses;
using GooseTape.NeuralNetwork.Maths;

namespace GooseTape.NeuralNetwork.Training;

/// <summary>
/// An immutable feed forward network trained by batch gradient descent.
/// </summary>
/// <remarks>
/// Training returns a new network rather than mutating this one. That is what makes an epoch
/// safe to retry: a failed or partially applied epoch leaves the previous network untouched,
/// so the orchestrator can resume from the last checkpoint without reasoning about torn state.
/// </remarks>
public sealed class FeedForwardNetwork
{
    private readonly LayerCollection _layers;
    private readonly ILossFunction _loss;

    private FeedForwardNetwork(LayerCollection layers, ILossFunction loss)
    {
        _layers = layers;
        _loss = loss;
    }

    /// <summary>Gets the layers of this network, from input side to output side.</summary>
    public LayerCollection Layers => _layers;

    /// <summary>Gets the loss function this network is trained against.</summary>
    public ILossFunction Loss => _loss;

    /// <summary>
    /// Creates a network from existing layers.
    /// </summary>
    /// <param name="layers">The layers, ordered from input side to output side.</param>
    /// <param name="loss">The loss function to train against.</param>
    /// <returns>A new <see cref="FeedForwardNetwork"/>.</returns>
    /// <exception cref="ArgumentNullException">Thrown when either argument is null.</exception>
    public static FeedForwardNetwork Create(LayerCollection layers, ILossFunction loss)
    {
        ArgumentNullException.ThrowIfNull(layers);
        ArgumentNullException.ThrowIfNull(loss);

        return new FeedForwardNetwork(layers, loss);
    }

    /// <summary>
    /// Runs a batch forward and returns the output distributions.
    /// </summary>
    /// <param name="input">The batch input, of shape <c>batch x inputs</c>.</param>
    /// <returns>The predicted distributions, of shape <c>batch x classes</c>.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="input"/> is null.</exception>
    public Matrix Predict(Matrix input)
    {
        ArgumentNullException.ThrowIfNull(input);

        return _layers.Forward(input).Output;
    }

    /// <summary>
    /// Performs one gradient descent step over a batch.
    /// </summary>
    /// <param name="input">The batch input, of shape <c>batch x inputs</c>.</param>
    /// <param name="expected">The one-hot expected outputs, of shape <c>batch x classes</c>.</param>
    /// <param name="learningRate">The step size to descend by.</param>
    /// <returns>The updated network, and the metrics measured before the update.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="input"/> or <paramref name="expected"/> is null.</exception>
    public BatchOutcome TrainOnBatch(Matrix input, Matrix expected, LearningRate learningRate)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(expected);

        var trace = _layers.Forward(input);
        var metrics = MetricsFor(trace.Output, expected);
        var gradients = ComputeGradients(trace, expected);

        return new BatchOutcome(
            new FeedForwardNetwork(_layers.Descend(gradients, learningRate.Value), _loss),
            metrics);
    }

    /// <summary>
    /// Measures mean loss and classification accuracy over a batch, without training.
    /// </summary>
    /// <param name="input">The batch input, of shape <c>batch x inputs</c>.</param>
    /// <param name="expected">The one-hot expected outputs, of shape <c>batch x classes</c>.</param>
    /// <returns>The measured metrics.</returns>
    /// <exception cref="ArgumentNullException">Thrown when either argument is null.</exception>
    public TrainingMetrics Evaluate(Matrix input, Matrix expected)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(expected);

        return MetricsFor(Predict(input), expected);
    }

    private TrainingMetrics MetricsFor(Matrix prediction, Matrix expected) => TrainingMetrics.Create(
        _loss.Compute(prediction, expected),
        CountCorrect(prediction, expected) / (double)prediction.Shape.RowCount,
        prediction.Shape.RowCount);

    private LayerGradient[] ComputeGradients(ForwardTrace trace, Matrix expected)
    {
        var gradients = new LayerGradient[_layers.Count];
        var delta = _loss.DeriveOutputDelta(trace.Output, expected);

        for (var index = _layers.Count - 1; index >= 1; index--)
        {
            gradients[index] = LayerGradient.From(trace[index], delta);
            delta = PropagateToPreviousLayer(index, delta, trace);
        }

        gradients[0] = LayerGradient.From(trace[0], delta);

        return gradients;
    }

    private Matrix PropagateToPreviousLayer(int index, Matrix delta, ForwardTrace trace) =>
        _layers[index - 1].DeriveWeightedInputDelta(
            _layers[index].Parameters.PropagateDelta(delta),
            trace[index - 1].Activation);

    private static int CountCorrect(Matrix prediction, Matrix expected)
    {
        var correct = 0;

        for (var row = 0; row < prediction.Shape.RowCount; row++)
        {
            correct += prediction.IndexOfLargestInRow(row) == expected.IndexOfLargestInRow(row) ? 1 : 0;
        }

        return correct;
    }
}
