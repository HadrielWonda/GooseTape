namespace GooseTape.NeuralNetwork.Maths;

/// <summary>
/// An immutable, row-major matrix of double precision values.
/// </summary>
/// <remarks>
/// <para>
/// Every operation returns a new instance, so matrices are safe to share across threads and
/// safe to capture in a checkpoint without defensive copying.
/// </para>
/// <para>
/// Performance: immutability costs one allocation per operation. For the network sizes this
/// project targets (784 x 128 x 10) the allocation pressure is acceptable and is dominated by
/// the cost of the multiplication itself. Replace with a pooled buffer strategy before scaling
/// to convolutional topologies.
/// </para>
/// </remarks>
public sealed class Matrix
{
    private readonly MatrixShape _shape;
    private readonly double[] _values;

    private Matrix(MatrixShape shape, double[] values)
    {
        _shape = shape;
        _values = values;
    }

    /// <summary>Gets the dimensions of this matrix.</summary>
    public MatrixShape Shape => _shape;

    /// <summary>
    /// Gets the row-major backing buffer without copying, for use by arithmetic inside this assembly.
    /// </summary>
    /// <remarks>
    /// Exposed internally so the operation classes can avoid per-element index arithmetic and the
    /// defensive copy that <see cref="ToArray"/> performs. The span is read-only, so immutability
    /// of the public surface is preserved.
    /// </remarks>
    internal ReadOnlySpan<double> Values => _values;

    /// <summary>
    /// Adopts a caller-owned buffer as the backing store of a new matrix, without copying.
    /// </summary>
    /// <param name="shape">The dimensions the buffer describes.</param>
    /// <param name="values">
    /// A buffer the caller has just allocated and will not retain. Ownership transfers to the matrix.
    /// </param>
    /// <returns>A new <see cref="Matrix"/> wrapping <paramref name="values"/>.</returns>
    internal static Matrix Wrap(MatrixShape shape, double[] values) => new(shape, values);

    /// <summary>Gets the value at the supplied zero based position.</summary>
    /// <param name="rowIndex">The zero based row index.</param>
    /// <param name="columnIndex">The zero based column index.</param>
    /// <returns>The value stored at that position.</returns>
    public double this[int rowIndex, int columnIndex] => _values[(rowIndex * _shape.ColumnCount) + columnIndex];

    /// <summary>
    /// Creates a matrix from a row-major value buffer.
    /// </summary>
    /// <param name="shape">The dimensions the buffer describes.</param>
    /// <param name="values">The row-major values. The buffer is copied.</param>
    /// <returns>A new <see cref="Matrix"/>.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="values"/> is null.</exception>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="values"/> does not contain exactly
    /// <see cref="MatrixShape.ElementCount"/> entries.
    /// </exception>
    public static Matrix FromValues(MatrixShape shape, IReadOnlyCollection<double> values)
    {
        ArgumentNullException.ThrowIfNull(values);

        if (values.Count != shape.ElementCount)
        {
            throw new ArgumentException(
                $"A {shape} matrix requires exactly {shape.ElementCount} values. Received {values.Count}.",
                nameof(values));
        }

        return new Matrix(shape, [.. values]);
    }

    /// <summary>Creates a matrix of the given shape with every element set to zero.</summary>
    /// <param name="shape">The dimensions of the new matrix.</param>
    /// <returns>A zero filled <see cref="Matrix"/>.</returns>
    public static Matrix Zeros(MatrixShape shape) => new(shape, new double[shape.ElementCount]);

    /// <summary>
    /// Creates a single row matrix from the supplied values.
    /// </summary>
    /// <param name="values">The values forming the row.</param>
    /// <returns>A one row <see cref="Matrix"/>.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="values"/> is null.</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="values"/> is empty.</exception>
    public static Matrix FromRow(IReadOnlyCollection<double> values)
    {
        ArgumentNullException.ThrowIfNull(values);

        if (values.Count == 0)
        {
            throw new ArgumentException("A row matrix requires at least one value.", nameof(values));
        }

        return FromValues(MatrixShape.Create(1, values.Count), values);
    }

    /// <summary>
    /// Creates a matrix whose elements are produced by <paramref name="generator"/>.
    /// </summary>
    /// <param name="shape">The dimensions of the new matrix.</param>
    /// <param name="generator">Produces the value for each element in row-major order.</param>
    /// <returns>A newly generated <see cref="Matrix"/>.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="generator"/> is null.</exception>
    public static Matrix Generate(MatrixShape shape, Func<double> generator)
    {
        ArgumentNullException.ThrowIfNull(generator);

        var values = new double[shape.ElementCount];

        for (var index = 0; index < values.Length; index++)
        {
            values[index] = generator();
        }

        return new Matrix(shape, values);
    }

    /// <summary>Copies this matrix into a new row-major array.</summary>
    /// <returns>A row-major copy of the underlying values.</returns>
    public double[] ToArray() => [.. _values];

    /// <summary>
    /// Applies <paramref name="transform"/> to every element.
    /// </summary>
    /// <param name="transform">The element-wise transform.</param>
    /// <returns>A new matrix of the same shape holding the transformed values.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="transform"/> is null.</exception>
    public Matrix Map(Func<double, double> transform)
    {
        ArgumentNullException.ThrowIfNull(transform);

        var values = new double[_values.Length];

        for (var index = 0; index < _values.Length; index++)
        {
            values[index] = transform(_values[index]);
        }

        return new Matrix(_shape, values);
    }

    /// <summary>Reads a single row as a new one row matrix.</summary>
    /// <param name="rowIndex">The zero based row to read.</param>
    /// <returns>A one row <see cref="Matrix"/> containing a copy of that row.</returns>
    /// <exception cref="ArgumentOutOfRangeException">
    /// Thrown when <paramref name="rowIndex"/> falls outside the matrix.
    /// </exception>
    public Matrix Row(int rowIndex)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(rowIndex);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(rowIndex, _shape.RowCount);

        var row = new double[_shape.ColumnCount];
        Array.Copy(_values, rowIndex * _shape.ColumnCount, row, 0, _shape.ColumnCount);

        return new Matrix(MatrixShape.Create(1, _shape.ColumnCount), row);
    }

    /// <summary>Returns the zero based column index holding the largest value in the given row.</summary>
    /// <param name="rowIndex">The zero based row to scan.</param>
    /// <returns>The column index of the largest value in that row.</returns>
    public int IndexOfLargestInRow(int rowIndex)
    {
        var offset = rowIndex * _shape.ColumnCount;
        var largestIndex = 0;

        for (var column = 1; column < _shape.ColumnCount; column++)
        {
            largestIndex = _values[offset + column] > _values[offset + largestIndex] ? column : largestIndex;
        }

        return largestIndex;
    }

    /// <summary>Returns the sum of every element.</summary>
    /// <returns>The total of all values in the matrix.</returns>
    public double Sum()
    {
        var total = 0d;

        foreach (var value in _values)
        {
            total += value;
        }

        return total;
    }
}
