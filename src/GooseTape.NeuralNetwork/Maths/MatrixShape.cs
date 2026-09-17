namespace GooseTape.NeuralNetwork.Maths;

/// <summary>
/// The dimensions of a <see cref="Matrix"/>, expressed as a row and column count.
/// </summary>
/// <remarks>
/// Wrapping the two dimensions in a single value object keeps <see cref="Matrix"/> to two
/// instance fields and makes dimension mismatches expressible as a single comparison.
/// </remarks>
/// <param name="RowCount">The number of rows. Must be greater than zero.</param>
/// <param name="ColumnCount">The number of columns. Must be greater than zero.</param>
public readonly record struct MatrixShape(int RowCount, int ColumnCount)
{
    /// <summary>
    /// Creates a validated shape.
    /// </summary>
    /// <param name="rowCount">The number of rows. Must be greater than zero.</param>
    /// <param name="columnCount">The number of columns. Must be greater than zero.</param>
    /// <returns>A validated <see cref="MatrixShape"/>.</returns>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when <paramref name="rowCount"/> or <paramref name="columnCount"/> is not positive.
    /// </exception>
    public static MatrixShape Create(int rowCount, int columnCount)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(rowCount, 1, nameof(rowCount));
        ArgumentOutOfRangeException.ThrowIfLessThan(columnCount, 1, nameof(columnCount));

        return new MatrixShape(rowCount, columnCount);
    }

    /// <summary>Gets the total number of elements described by this shape.</summary>
    public int ElementCount => RowCount * ColumnCount;

    /// <summary>Gets this shape with its rows and columns exchanged.</summary>
    public MatrixShape Transposed() => new(ColumnCount, RowCount);

    /// <summary>
    /// Determines whether a matrix of this shape can be multiplied by a matrix of <paramref name="other"/>.
    /// </summary>
    /// <param name="other">The shape of the right-hand operand.</param>
    /// <returns><see langword="true"/> when the inner dimensions agree.</returns>
    public bool CanMultiplyBy(MatrixShape other) => ColumnCount == other.RowCount;

    /// <summary>Returns a human readable representation such as <c>784x128</c>.</summary>
    /// <returns>The shape rendered as <c>rows x columns</c>.</returns>
    public override string ToString() => $"{RowCount}x{ColumnCount}";
}
