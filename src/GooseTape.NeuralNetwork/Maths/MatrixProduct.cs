namespace GooseTape.NeuralNetwork.Maths;

/// <summary>
/// Matrix products and transposition.
/// </summary>
public static class MatrixProduct
{
    /// <summary>
    /// Multiplies <paramref name="left"/> by <paramref name="right"/>.
    /// </summary>
    /// <param name="left">The left operand, of shape <c>m x n</c>.</param>
    /// <param name="right">The right operand, of shape <c>n x p</c>.</param>
    /// <returns>The product, of shape <c>m x p</c>.</returns>
    /// <exception cref="ArgumentNullException">Thrown when either operand is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the inner dimensions disagree.</exception>
    public static Matrix Multiply(this Matrix left, Matrix right)
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);

        if (!left.Shape.CanMultiplyBy(right.Shape))
        {
            throw new ArgumentException(
                $"Cannot multiply a {left.Shape} matrix by a {right.Shape} matrix: inner dimensions must agree.",
                nameof(right));
        }

        var shape = MatrixShape.Create(left.Shape.RowCount, right.Shape.ColumnCount);
        var values = new double[shape.ElementCount];

        MultiplyInto(left, right, values);

        return Matrix.Wrap(shape, values);
    }

    /// <summary>
    /// Returns a new matrix with the rows and columns of <paramref name="source"/> exchanged.
    /// </summary>
    /// <param name="source">The matrix to transpose.</param>
    /// <returns>The transposed matrix.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="source"/> is null.</exception>
    public static Matrix Transpose(this Matrix source)
    {
        ArgumentNullException.ThrowIfNull(source);

        var shape = source.Shape.Transposed();
        var values = new double[shape.ElementCount];

        TransposeInto(source, values);

        return Matrix.Wrap(shape, values);
    }

    private static void MultiplyInto(Matrix left, Matrix right, double[] destination)
    {
        var innerCount = left.Shape.ColumnCount;
        var outputColumnCount = right.Shape.ColumnCount;

        for (var row = 0; row < left.Shape.RowCount; row++)
        {
            AccumulateRow(left, right, destination, row, innerCount, outputColumnCount);
        }
    }

    private static void AccumulateRow(
        Matrix left,
        Matrix right,
        double[] destination,
        int row,
        int innerCount,
        int outputColumnCount)
    {
        var leftValues = left.Values;
        var rightValues = right.Values;
        var destinationOffset = row * outputColumnCount;
        var leftOffset = row * innerCount;

        for (var inner = 0; inner < innerCount; inner++)
        {
            AccumulateScaledRow(
                rightValues.Slice(inner * outputColumnCount, outputColumnCount),
                leftValues[leftOffset + inner],
                destination.AsSpan(destinationOffset, outputColumnCount));
        }
    }

    private static void AccumulateScaledRow(ReadOnlySpan<double> source, double scale, Span<double> destination)
    {
        for (var index = 0; index < destination.Length; index++)
        {
            destination[index] += source[index] * scale;
        }
    }

    private static void TransposeInto(Matrix source, double[] destination)
    {
        var rowCount = source.Shape.RowCount;

        for (var index = 0; index < destination.Length; index++)
        {
            destination[index] = source[index % rowCount, index / rowCount];
        }
    }
}
