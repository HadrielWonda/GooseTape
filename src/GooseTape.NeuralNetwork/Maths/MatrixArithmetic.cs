namespace GooseTape.NeuralNetwork.Maths;

/// <summary>
/// Element-wise matrix arithmetic and the broadcasting helpers the dense layers rely on.
/// </summary>
public static class MatrixArithmetic
{
    /// <summary>Adds two matrices of identical shape.</summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The element-wise sum.</returns>
    /// <exception cref="ArgumentNullException">Thrown when either operand is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the shapes differ.</exception>
    public static Matrix Add(this Matrix left, Matrix right) => Combine(left, right, static (a, b) => a + b, "add");

    /// <summary>Subtracts <paramref name="right"/> from <paramref name="left"/>.</summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The element-wise difference.</returns>
    /// <exception cref="ArgumentNullException">Thrown when either operand is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the shapes differ.</exception>
    public static Matrix Subtract(this Matrix left, Matrix right) =>
        Combine(left, right, static (a, b) => a - b, "subtract");

    /// <summary>Multiplies two matrices element by element.</summary>
    /// <param name="left">The left operand.</param>
    /// <param name="right">The right operand.</param>
    /// <returns>The Hadamard product.</returns>
    /// <exception cref="ArgumentNullException">Thrown when either operand is null.</exception>
    /// <exception cref="ArgumentException">Thrown when the shapes differ.</exception>
    public static Matrix HadamardProduct(this Matrix left, Matrix right) =>
        Combine(left, right, static (a, b) => a * b, "multiply element-wise");

    /// <summary>Multiplies every element by <paramref name="factor"/>.</summary>
    /// <param name="source">The matrix to scale.</param>
    /// <param name="factor">The scalar factor.</param>
    /// <returns>The scaled matrix.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="source"/> is null.</exception>
    public static Matrix Scale(this Matrix source, double factor)
    {
        ArgumentNullException.ThrowIfNull(source);

        // Deliberately not Map(value => value * factor): that lambda captures `factor`, so every
        // call allocates a closure and invokes a delegate per element. Scaling runs on
        // gradient-sized matrices for every batch, which makes it worth the explicit loop.
        var values = new double[source.Shape.ElementCount];

        for (var index = 0; index < values.Length; index++)
        {
            values[index] = source.Values[index] * factor;
        }

        return Matrix.Wrap(source.Shape, values);
    }

    /// <summary>
    /// Adds a single row <paramref name="rowVector"/> to every row of <paramref name="source"/>.
    /// </summary>
    /// <param name="source">The matrix whose rows receive the addition, of shape <c>batch x n</c>.</param>
    /// <param name="rowVector">The row to broadcast, of shape <c>1 x n</c>.</param>
    /// <returns>A matrix of the same shape as <paramref name="source"/>.</returns>
    /// <exception cref="ArgumentNullException">Thrown when either operand is null.</exception>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="rowVector"/> is not a single row, or its width does not match.
    /// </exception>
    public static Matrix AddRowVector(this Matrix source, Matrix rowVector)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(rowVector);
        GuardBroadcastable(source.Shape, rowVector.Shape);

        var values = new double[source.Shape.ElementCount];
        var columnCount = source.Shape.ColumnCount;

        for (var index = 0; index < values.Length; index++)
        {
            values[index] = source.Values[index] + rowVector.Values[index % columnCount];
        }

        return Matrix.Wrap(source.Shape, values);
    }

    /// <summary>
    /// Collapses every row of <paramref name="source"/> into a single row holding the column totals.
    /// </summary>
    /// <param name="source">The matrix to reduce, of shape <c>batch x n</c>.</param>
    /// <returns>A matrix of shape <c>1 x n</c> holding the per column sums.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="source"/> is null.</exception>
    public static Matrix SumRows(this Matrix source)
    {
        ArgumentNullException.ThrowIfNull(source);

        var columnCount = source.Shape.ColumnCount;
        var values = new double[columnCount];

        for (var index = 0; index < source.Shape.ElementCount; index++)
        {
            values[index % columnCount] += source.Values[index];
        }

        return Matrix.Wrap(MatrixShape.Create(1, columnCount), values);
    }

    private static Matrix Combine(Matrix left, Matrix right, Func<double, double, double> combine, string operationName)
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);
        GuardSameShape(left.Shape, right.Shape, operationName);

        var values = new double[left.Shape.ElementCount];

        for (var index = 0; index < values.Length; index++)
        {
            values[index] = combine(left.Values[index], right.Values[index]);
        }

        return Matrix.Wrap(left.Shape, values);
    }

    private static void GuardSameShape(MatrixShape left, MatrixShape right, string operationName)
    {
        if (left == right)
        {
            return;
        }

        throw new ArgumentException(
            $"Cannot {operationName} a {left} matrix and a {right} matrix: shapes must be identical.",
            nameof(right));
    }

    private static void GuardBroadcastable(MatrixShape source, MatrixShape rowVector)
    {
        if (rowVector.RowCount == 1 && rowVector.ColumnCount == source.ColumnCount)
        {
            return;
        }

        throw new ArgumentException(
            $"Cannot broadcast a {rowVector} matrix across a {source} matrix: expected a 1x{source.ColumnCount} row.",
            nameof(rowVector));
    }
}
