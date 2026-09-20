using GooseTape.NeuralNetwork.Activations;
using GooseTape.NeuralNetwork.Losses;
using GooseTape.NeuralNetwork.Maths;

namespace GooseTape.NeuralNetwork.Tests;

public sealed class ActivationAndLossTests
{
    [Fact]
    public void Softmax_ProducesRowsThatSumToOne()
    {
        var logits = Matrix.FromValues(MatrixShape.Create(2, 3), [1, 2, 3, -1, 0, 1]);

        var probabilities = new SoftmaxActivation().Activate(logits);

        probabilities.Row(0).Sum().Should().BeApproximately(1d, 1e-12);
        probabilities.Row(1).Sum().Should().BeApproximately(1d, 1e-12);
    }

    [Fact]
    public void Softmax_WithLogitsThatWouldOverflow_StaysFinite()
    {
        // Without the per-row maximum subtraction, exp(1000) overflows to infinity and every
        // probability becomes NaN.
        var logits = Matrix.FromRow([1000d, 1001d, 1002d]);

        var probabilities = new SoftmaxActivation().Activate(logits);

        probabilities.ToArray().Should().OnlyContain(value => double.IsFinite(value));
        probabilities.Sum().Should().BeApproximately(1d, 1e-12);
        probabilities.IndexOfLargestInRow(0).Should().Be(2);
    }

    [Fact]
    public void Softmax_WithVeryNegativeLogits_StaysFiniteAndNormalised()
    {
        // Every exponent underflows towards zero here, so a naive implementation divides by zero.
        var logits = Matrix.FromRow([-1000d, -1001d, -1002d]);

        var probabilities = new SoftmaxActivation().Activate(logits);

        probabilities.ToArray().Should().OnlyContain(value => double.IsFinite(value));
        probabilities.Sum().Should().BeApproximately(1d, 1e-12);
        probabilities.IndexOfLargestInRow(0).Should().Be(0);
    }

    [Fact]
    public void Softmax_WithEqualLogits_SpreadsProbabilityEvenly()
    {
        var probabilities = new SoftmaxActivation().Activate(Matrix.FromRow([4d, 4d, 4d, 4d]));

        probabilities.ToArray().Should().AllSatisfy(value => value.Should().BeApproximately(0.25d, 1e-12));
    }

    [Fact]
    public void Softmax_WithASingleClass_ReturnsCertainty()
    {
        var probabilities = new SoftmaxActivation().Activate(Matrix.FromRow([-7.5d]));

        probabilities[0, 0].Should().BeApproximately(1d, 1e-12);
    }

    [Fact]
    public void Softmax_IsShiftInvariant()
    {
        var activation = new SoftmaxActivation();
        var logits = Matrix.FromRow([0.5, -1.25, 2d]);
        var shifted = logits.Map(static value => value + 17d);

        var original = activation.Activate(logits).ToArray();
        var afterShift = activation.Activate(shifted).ToArray();

        afterShift.Should().BeEquivalentTo(original, options => options.Using<double>(
            context => context.Subject.Should().BeApproximately(context.Expectation, 1e-12)).WhenTypeIs<double>());
    }

    [Theory]
    [InlineData(-2d, 0d)]
    [InlineData(0d, 0d)]
    [InlineData(3.5d, 3.5d)]
    public void RectifiedLinear_ClampsNegativesToZero(double input, double expected)
    {
        new RectifiedLinearActivation().Activate(Matrix.FromRow([input]))[0, 0].Should().Be(expected);
    }

    [Fact]
    public void RectifiedLinear_DerivativeIsOneWhereActive()
    {
        var activation = Matrix.FromRow([0d, 0.5d, 2d]);

        new RectifiedLinearActivation().DeriveFromActivation(activation).ToArray().Should().Equal(0, 1, 1);
    }

    [Fact]
    public void CrossEntropy_ForAConfidentCorrectPrediction_IsNearZero()
    {
        var prediction = Matrix.FromRow([0.999, 0.0005, 0.0005]);
        var expected = Matrix.FromRow([1, 0, 0]);

        new CrossEntropyLoss().Compute(prediction, expected).Should().BeLessThan(0.002);
    }

    [Fact]
    public void CrossEntropy_ForAConfidentWrongPrediction_IsLarge()
    {
        var prediction = Matrix.FromRow([0.0005, 0.999, 0.0005]);
        var expected = Matrix.FromRow([1, 0, 0]);

        new CrossEntropyLoss().Compute(prediction, expected).Should().BeGreaterThan(7d);
    }

    [Fact]
    public void CrossEntropy_WithACertainAndWrongPrediction_StaysFiniteRatherThanInfinite()
    {
        // log(0) is negative infinity; the probability floor is what keeps a run from poisoning
        // every subsequent metric with NaN.
        var prediction = Matrix.FromRow([0d, 1d]);
        var expected = Matrix.FromRow([1d, 0d]);

        double.IsFinite(new CrossEntropyLoss().Compute(prediction, expected)).Should().BeTrue();
    }

    [Fact]
    public void CrossEntropy_OutputDelta_IsPredictionMinusExpected()
    {
        var prediction = Matrix.FromRow([0.7, 0.2, 0.1]);
        var expected = Matrix.FromRow([1, 0, 0]);

        var delta = new CrossEntropyLoss().DeriveOutputDelta(prediction, expected);

        delta[0, 0].Should().BeApproximately(-0.3d, 1e-12);
        delta[0, 1].Should().BeApproximately(0.2d, 1e-12);
        delta[0, 2].Should().BeApproximately(0.1d, 1e-12);
    }
}
