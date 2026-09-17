using GooseTape.NeuralNetwork.Maths;

namespace GooseTape.NeuralNetwork.Tests;

public sealed class MatrixTests
{
    [Fact]
    public void Multiply_WithAgreeingInnerDimensions_ProducesExpectedProduct()
    {
        // Arrange
        var left = Matrix.FromValues(MatrixShape.Create(2, 3), [1, 2, 3, 4, 5, 6]);
        var right = Matrix.FromValues(MatrixShape.Create(3, 2), [7, 8, 9, 10, 11, 12]);

        // Act
        var product = left.Multiply(right);

        // Assert
        product.Shape.Should().Be(MatrixShape.Create(2, 2));
        product[0, 0].Should().Be(58);
        product[0, 1].Should().Be(64);
        product[1, 0].Should().Be(139);
        product[1, 1].Should().Be(154);
    }

    [Fact]
    public void Multiply_WithDisagreeingInnerDimensions_ThrowsDescribingBothShapes()
    {
        var left = Matrix.Zeros(MatrixShape.Create(2, 3));
        var right = Matrix.Zeros(MatrixShape.Create(4, 2));

        var act = () => left.Multiply(right);

        act.Should().Throw<ArgumentException>().WithMessage("*2x3*4x2*");
    }

    [Fact]
    public void Transpose_ExchangesRowsAndColumns()
    {
        var source = Matrix.FromValues(MatrixShape.Create(2, 3), [1, 2, 3, 4, 5, 6]);

        var transposed = source.Transpose();

        transposed.Shape.Should().Be(MatrixShape.Create(3, 2));
        transposed.ToArray().Should().Equal(1, 4, 2, 5, 3, 6);
    }

    [Fact]
    public void Transpose_AppliedTwice_ReturnsTheOriginal()
    {
        var source = Matrix.FromValues(MatrixShape.Create(3, 4), [.. Enumerable.Range(0, 12).Select(value => (double)value)]);

        source.Transpose().Transpose().ToArray().Should().Equal(source.ToArray());
    }

    [Fact]
    public void AddRowVector_AddsTheSameRowToEveryRow()
    {
        var source = Matrix.FromValues(MatrixShape.Create(2, 3), [1, 1, 1, 2, 2, 2]);
        var bias = Matrix.FromRow([10, 20, 30]);

        source.AddRowVector(bias).ToArray().Should().Equal(11, 21, 31, 12, 22, 32);
    }

    [Fact]
    public void SumRows_CollapsesToPerColumnTotals()
    {
        var source = Matrix.FromValues(MatrixShape.Create(3, 2), [1, 2, 3, 4, 5, 6]);

        source.SumRows().ToArray().Should().Equal(9, 12);
    }

    [Fact]
    public void FromValues_WithWrongValueCount_ThrowsNamingTheExpectedCount()
    {
        var act = () => Matrix.FromValues(MatrixShape.Create(2, 2), [1, 2, 3]);

        act.Should().Throw<ArgumentException>().WithMessage("*exactly 4 values*");
    }

    [Theory]
    [InlineData(0, 1)]
    [InlineData(1, 0)]
    [InlineData(-3, 4)]
    public void MatrixShape_WithNonPositiveDimension_IsRejected(int rowCount, int columnCount)
    {
        var act = () => MatrixShape.Create(rowCount, columnCount);

        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void IndexOfLargestInRow_FindsThePerRowMaximum()
    {
        var source = Matrix.FromValues(MatrixShape.Create(2, 3), [0.1, 0.7, 0.2, 0.9, 0.05, 0.05]);

        source.IndexOfLargestInRow(0).Should().Be(1);
        source.IndexOfLargestInRow(1).Should().Be(0);
    }
}
