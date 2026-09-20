using GooseTape.NeuralNetwork.Activations;
using GooseTape.NeuralNetwork.Checkpoints;
using GooseTape.NeuralNetwork.Data;
using GooseTape.NeuralNetwork.Layers;
using GooseTape.NeuralNetwork.Losses;
using GooseTape.NeuralNetwork.Maths;
using GooseTape.NeuralNetwork.Training;

namespace GooseTape.NeuralNetwork.Tests;

/// <summary>
/// Covers the fail-fast guards on the value objects and collections: the inputs each type
/// refuses, and the message it refuses them with.
/// </summary>
public sealed class ValidationTests
{
    [Theory]
    [InlineData(0d)]
    [InlineData(-0.5d)]
    [InlineData(1.5d)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void LearningRate_OutsideTheStableRange_IsRejected(double value)
    {
        var act = () => LearningRate.Create(value);

        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void LearningRate_AtTheUpperBound_IsAccepted()
    {
        LearningRate.Create(1d).Value.Should().Be(1d);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    [InlineData(8193)]
    public void BatchSize_OutsideTheSupportedRange_IsRejected(int value)
    {
        var act = () => BatchSize.Create(value);

        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void EpochNumber_BelowOne_IsRejected()
    {
        var act = () => EpochNumber.Create(0);

        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void EpochNumber_Next_Advances()
    {
        EpochNumber.Create(4).Next().Value.Should().Be(5);
    }

    [Theory]
    [InlineData(-0.1d, 0.5d, 10)]
    [InlineData(double.NaN, 0.5d, 10)]
    [InlineData(0.5d, 1.2d, 10)]
    [InlineData(0.5d, -0.1d, 10)]
    [InlineData(0.5d, 0.5d, 0)]
    public void TrainingMetrics_WithValuesOutsideTheirRanges_AreRejected(double loss, double accuracy, int examples)
    {
        var act = () => TrainingMetrics.Create(loss, accuracy, examples);

        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void MetricsAccumulator_WithNothingAccumulated_RefusesToReport()
    {
        var act = () => new TrainingMetricsAccumulator().ToMetrics();

        act.Should().Throw<InvalidOperationException>().WithMessage("*no metrics*");
    }

    [Fact]
    public void MetricsAccumulator_WeightsBatchesByTheirExampleCount()
    {
        // A 90-example batch at 100% and a 10-example batch at 0% average to 90%, not 50%.
        var accumulator = new TrainingMetricsAccumulator();
        accumulator.Add(TrainingMetrics.Create(0.1d, 1d, 90));
        accumulator.Add(TrainingMetrics.Create(1.1d, 0d, 10));

        var metrics = accumulator.ToMetrics();

        metrics.Accuracy.Should().BeApproximately(0.9d, 1e-12);
        metrics.MeanLoss.Should().BeApproximately(0.2d, 1e-12);
        metrics.ExampleCount.Should().Be(100);
    }

    [Fact]
    public void NetworkTopology_WithoutAHiddenLayer_IsRejected()
    {
        var act = () => NetworkTopology.Create([784, 10]);

        act.Should().Throw<ArgumentException>().WithMessage("*input size*hidden size*output size*");
    }

    [Fact]
    public void NetworkTopology_WithANonPositiveLayer_IsRejected()
    {
        var act = () => NetworkTopology.Create([784, 0, 10]);

        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void NetworkTopology_ReportsItsShape()
    {
        var topology = NetworkTopology.Create([784, 128, 32, 10]);

        topology.InputSize.Value.Should().Be(784);
        topology.OutputSize.Value.Should().Be(10);
        topology.DenseLayerCount.Should().Be(3);
        topology.Select(size => size.Value).Should().Equal(784, 128, 32, 10);
    }

    [Fact]
    public void ActivationRegistry_WithAnUnknownName_ListsWhatItKnows()
    {
        var act = () => ActivationRegistry.Default.Resolve("sigmoid");

        act.Should().Throw<NotSupportedException>().WithMessage("*sigmoid*relu*softmax*");
    }

    [Fact]
    public void LossRegistry_WithAnUnknownName_ListsWhatItKnows()
    {
        var act = () => LossRegistry.Default.Resolve("hinge");

        act.Should().Throw<NotSupportedException>().WithMessage("*hinge*cross-entropy*");
    }

    [Theory]
    [InlineData("")]
    [InlineData("   ")]
    public void Registries_WithABlankName_AreRejected(string name)
    {
        var resolveActivation = () => ActivationRegistry.Default.Resolve(name);
        var resolveLoss = () => LossRegistry.Default.Resolve(name);

        resolveActivation.Should().Throw<ArgumentException>();
        resolveLoss.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void EmptyRegistries_AreRejected()
    {
        var activations = () => ActivationRegistry.From(new Dictionary<string, Func<IActivationFunction>>(StringComparer.Ordinal));
        var losses = () => LossRegistry.From(new Dictionary<string, Func<ILossFunction>>(StringComparer.Ordinal));

        activations.Should().Throw<ArgumentException>();
        losses.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void LayerCollection_WithASingleLayer_IsRejected()
    {
        var layer = DenseLayer.Create(
            LayerParameters.Create(Matrix.Zeros(MatrixShape.Create(2, 2)), Matrix.Zeros(MatrixShape.Create(1, 2))),
            new RectifiedLinearActivation());

        var act = () => LayerCollection.From([layer]);

        act.Should().Throw<ArgumentException>().WithMessage("*at least one hidden layer*");
    }

    [Fact]
    public void LayerCollection_DescendingWithTheWrongGradientCount_IsRejected()
    {
        var network = NetworkInitializer.Create(NetworkTopology.Create([3, 4, 2]), RandomSeed.Create(1));
        var gradient = LayerGradient.From(
            new LayerForwardPass(Matrix.Zeros(MatrixShape.Create(1, 3)), Matrix.Zeros(MatrixShape.Create(1, 4))),
            Matrix.Zeros(MatrixShape.Create(1, 4)));

        var act = () => network.Layers.Descend([gradient], 0.1d);

        act.Should().Throw<ArgumentException>().WithMessage("*Expected 2 gradients*");
    }

    [Fact]
    public void ForwardTrace_WithNoPasses_IsRejected()
    {
        var act = () => ForwardTrace.From([]);

        act.Should().Throw<ArgumentException>().WithMessage("*at least one layer pass*");
    }

    [Fact]
    public void LayerParameters_WithBiasesThatDoNotMatchTheWeights_AreRejected()
    {
        var act = () => LayerParameters.Create(
            Matrix.Zeros(MatrixShape.Create(3, 4)),
            Matrix.Zeros(MatrixShape.Create(1, 5)));

        act.Should().Throw<ArgumentException>().WithMessage("*1x4*");
    }

    [Fact]
    public void SoftmaxAsAHiddenLayer_IsRefusedRatherThanSilentlyWrong()
    {
        // Softmax has no standalone derivative, so it is only valid on the output layer.
        var layer = DenseLayer.Create(
            LayerParameters.Create(Matrix.Zeros(MatrixShape.Create(2, 2)), Matrix.Zeros(MatrixShape.Create(1, 2))),
            new SoftmaxActivation());

        var act = () => layer.DeriveWeightedInputDelta(
            Matrix.Zeros(MatrixShape.Create(1, 2)),
            Matrix.Zeros(MatrixShape.Create(1, 2)));

        act.Should().Throw<InvalidOperationException>().WithMessage("*softmax*output layer*");
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(10)]
    public void DigitLabel_OutsideZeroToNine_IsRejected(int value)
    {
        var act = () => DigitLabel.Create(value);

        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void DigitLabel_OneHotIntoTheWrongWidth_IsRejected()
    {
        var destination = new double[3];

        try
        {
            DigitLabel.Create(4).WriteOneHot(destination);
            Assert.Fail("Expected an ArgumentException.");
        }
        catch (ArgumentException exception)
        {
            exception.Message.Should().Contain("10 wide");
        }
    }

    [Theory]
    [InlineData(-0.1d)]
    [InlineData(1.1d)]
    [InlineData(double.NaN)]
    public void PixelGrid_WithIntensitiesOutsideZeroToOne_IsRejected(double intensity)
    {
        var act = () => PixelGrid.FromNormalised([0.5d, intensity]);

        act.Should().Throw<ArgumentException>().WithMessage("*within [0, 1]*");
    }

    [Fact]
    public void PixelGrid_WithNoPixels_IsRejected()
    {
        var fromNormalised = () => PixelGrid.FromNormalised([]);
        var fromGreyscale = () => PixelGrid.FromGreyscale(ReadOnlySpan<byte>.Empty);

        fromNormalised.Should().Throw<ArgumentException>();
        fromGreyscale.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void ImageDataset_WithImagesOfDifferentSizes_IsRejected()
    {
        var wide = LabelledImage.Create(PixelGrid.FromNormalised([0.1d, 0.2d, 0.3d]), DigitLabel.Create(1));
        var narrow = LabelledImage.Create(PixelGrid.FromNormalised([0.1d, 0.2d]), DigitLabel.Create(2));

        var act = () => ImageDataset.From([wide, narrow]);

        act.Should().Throw<ArgumentException>().WithMessage("*must carry 3 pixels*");
    }

    [Fact]
    public void ImageDataset_WithNoImages_IsRejected()
    {
        var act = () => ImageDataset.From([]);

        act.Should().Throw<ArgumentException>().WithMessage("*at least one image*");
    }

    [Fact]
    public void ImageDataset_BatchesCoverEveryExampleIncludingAShortFinalBatch()
    {
        var dataset = SyntheticDatasetSource.Create(1, 5).Build();   // 10 images, one per class

        var batches = dataset.Batches(BatchSize.Create(4)).ToList();

        batches.Should().HaveCount(3);
        batches.Select(batch => batch.Inputs.Shape.RowCount).Should().Equal(4, 4, 2);
        batches.Sum(batch => batch.Inputs.Shape.RowCount).Should().Be(dataset.Count);
    }

    [Fact]
    public void ImageDataset_TakeAtMost_IsBoundedAndValidated()
    {
        var dataset = SyntheticDatasetSource.Create(2, 7).Build();   // 20 images

        dataset.TakeAtMost(5).Count.Should().Be(5);
        dataset.TakeAtMost(500).Count.Should().Be(20);
        ((Action)(() => dataset.TakeAtMost(0))).Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void CheckpointNotFound_NamesTheMissingCheckpoint()
    {
        var byId = new CheckpointNotFoundException(CheckpointId.Create("run-7-epoch-2"));
        var byMessage = new CheckpointNotFoundException("custom message");
        var wrapped = new CheckpointNotFoundException("wrapped", new InvalidOperationException("inner"));

        byId.Message.Should().Contain("run-7-epoch-2");
        byId.CheckpointId.Value.Should().Be("run-7-epoch-2");
        byMessage.Message.Should().Be("custom message");
        wrapped.InnerException.Should().BeOfType<InvalidOperationException>();
        new CheckpointNotFoundException().Message.Should().NotBeNullOrWhiteSpace();
    }

    [Fact]
    public void NetworkSnapshot_WithNoLayers_IsRejected()
    {
        var act = () => NetworkSerializer.Default.FromSnapshot(new NetworkSnapshot([], "cross-entropy"));

        act.Should().Throw<InvalidDataException>().WithMessage("*at least one layer*");
    }

    [Fact]
    public void NetworkSnapshot_NamingAnUnknownLoss_IsRejected()
    {
        var snapshot = new NetworkSnapshot(
            [
                new LayerSnapshot(2, 2, "relu", [1, 2, 3, 4], [0, 0]),
                new LayerSnapshot(2, 2, "softmax", [1, 2, 3, 4], [0, 0]),
            ],
            "hinge");

        var act = () => NetworkSerializer.Default.FromSnapshot(snapshot);

        act.Should().Throw<NotSupportedException>().WithMessage("*hinge*");
    }
}
