using BenchmarkDotNet.Attributes;
using GooseTape.NeuralNetwork.Activations;
using GooseTape.NeuralNetwork.Maths;

namespace GooseTape.Benchmarks;

/// <summary>
/// Measures the matrix operations the training loop spends its time in, at MNIST shapes.
/// </summary>
/// <remarks>
/// The shapes are the ones a 784 to 128 to 10 network actually uses, so the numbers say
/// something about training rather than about matrices in the abstract.
/// </remarks>
[MemoryDiagnoser]
public class MatrixBenchmarks
{
    private Matrix _batch = default!;
    private Matrix _weights = default!;
    private Matrix _hidden = default!;
    private Matrix _otherHidden = default!;

    /// <summary>Gets or sets the number of examples per batch.</summary>
    [Params(32, 64, 128)]
    public int BatchSize { get; set; }

    /// <summary>Builds the operands once per parameter set.</summary>
    [GlobalSetup]
    public void Setup()
    {
        var random = new Random(20_260_920);
        double Next() => (random.NextDouble() * 2d) - 1d;

        _batch = Matrix.Generate(MatrixShape.Create(BatchSize, 784), Next);
        _weights = Matrix.Generate(MatrixShape.Create(784, 128), Next);
        _hidden = Matrix.Generate(MatrixShape.Create(BatchSize, 128), Next);
        _otherHidden = Matrix.Generate(MatrixShape.Create(BatchSize, 128), Next);
    }

    /// <summary>The forward pass of the first dense layer, and the dominant cost of an epoch.</summary>
    [Benchmark(Baseline = true)]
    public Matrix MultiplyInputByWeights() => _batch.Multiply(_weights);

    /// <summary>Transposing the input, which the weight gradient needs on every batch.</summary>
    [Benchmark]
    public Matrix TransposeBatch() => _batch.Transpose();

    /// <summary>Scaling a gradient by the learning rate.</summary>
    [Benchmark]
    public Matrix ScaleHidden() => _hidden.Scale(0.1d);

    /// <summary>Adding two matrices, as a parameter update does.</summary>
    [Benchmark]
    public Matrix AddHidden() => _hidden.Add(_otherHidden);

    /// <summary>The element-wise product the backward pass applies to the activation derivative.</summary>
    [Benchmark]
    public Matrix HadamardHidden() => _hidden.HadamardProduct(_otherHidden);

    /// <summary>Collapsing a batch into per-neuron totals, which the bias gradient needs.</summary>
    [Benchmark]
    public Matrix SumRowsOfHidden() => _hidden.SumRows();

    /// <summary>A rectified linear activation, which goes through the delegate-based map.</summary>
    [Benchmark]
    public Matrix ActivateRectifiedLinear() => new RectifiedLinearActivation().Activate(_hidden);

    /// <summary>A softmax over the output layer, including its per-row maximum pass.</summary>
    [Benchmark]
    public Matrix ActivateSoftmax() => new SoftmaxActivation().Activate(_otherHidden);
}
