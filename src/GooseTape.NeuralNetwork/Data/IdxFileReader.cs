using System.Buffers.Binary;
using System.IO.Compression;

namespace GooseTape.NeuralNetwork.Data;

/// <summary>
/// Reads the IDX container format the MNIST distribution uses.
/// </summary>
/// <remarks>
/// IDX stores a big-endian magic number, then one big-endian dimension per axis, then the raw
/// payload. Files ending in <c>.gz</c> are decompressed transparently, because the canonical
/// MNIST download ships that way.
/// </remarks>
public static class IdxFileReader
{
    private const int ImageMagicNumber = 2051;
    private const int LabelMagicNumber = 2049;

    /// <summary>
    /// Reads an IDX image file into one flattened pixel grid per image.
    /// </summary>
    /// <param name="filePath">The path to the image file, optionally gzip compressed.</param>
    /// <param name="cancellationToken">Token to monitor for cancellation requests.</param>
    /// <returns>The pixel grids, in file order.</returns>
    /// <exception cref="ArgumentException">Thrown when <paramref name="filePath"/> is null or blank.</exception>
    /// <exception cref="FileNotFoundException">Thrown when the file does not exist.</exception>
    /// <exception cref="InvalidDataException">Thrown when the file is not an IDX image file.</exception>
    public static async Task<IReadOnlyList<PixelGrid>> ReadImagesAsync(
        string filePath,
        CancellationToken cancellationToken = default)
    {
        var payload = await ReadPayloadAsync(filePath, ImageMagicNumber, 3, cancellationToken).ConfigureAwait(false);
        var pixelsPerImage = payload.Dimensions[1] * payload.Dimensions[2];

        GuardPayloadLength(payload, payload.Dimensions[0] * pixelsPerImage, filePath);

        var images = new PixelGrid[payload.Dimensions[0]];

        for (var index = 0; index < images.Length; index++)
        {
            images[index] = PixelGrid.FromGreyscale(payload.Body.AsSpan(index * pixelsPerImage, pixelsPerImage));
        }

        return images;
    }

    /// <summary>
    /// Reads an IDX label file into digit labels.
    /// </summary>
    /// <param name="filePath">The path to the label file, optionally gzip compressed.</param>
    /// <param name="cancellationToken">Token to monitor for cancellation requests.</param>
    /// <returns>The labels, in file order.</returns>
    /// <exception cref="ArgumentException">Thrown when <paramref name="filePath"/> is null or blank.</exception>
    /// <exception cref="FileNotFoundException">Thrown when the file does not exist.</exception>
    /// <exception cref="InvalidDataException">Thrown when the file is not an IDX label file.</exception>
    public static async Task<IReadOnlyList<DigitLabel>> ReadLabelsAsync(
        string filePath,
        CancellationToken cancellationToken = default)
    {
        var payload = await ReadPayloadAsync(filePath, LabelMagicNumber, 1, cancellationToken).ConfigureAwait(false);

        GuardPayloadLength(payload, payload.Dimensions[0], filePath);

        return [.. payload.Body.Select(label => DigitLabel.Create(label))];
    }

    private static async Task<IdxPayload> ReadPayloadAsync(
        string filePath,
        int expectedMagicNumber,
        int expectedDimensionCount,
        CancellationToken cancellationToken)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(filePath);

        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException($"The IDX file {filePath} does not exist.", filePath);
        }

        var bytes = await ReadAllBytesAsync(filePath, cancellationToken).ConfigureAwait(false);
        var headerLength = sizeof(int) * (1 + expectedDimensionCount);

        GuardHeaderLength(bytes.Length, headerLength, filePath);
        GuardMagicNumber(ReadBigEndian(bytes, 0), expectedMagicNumber, filePath);

        var dimensions = new int[expectedDimensionCount];

        for (var axis = 0; axis < expectedDimensionCount; axis++)
        {
            dimensions[axis] = ReadBigEndian(bytes, sizeof(int) * (axis + 1));
        }

        return new IdxPayload(dimensions, bytes[headerLength..]);
    }

    private static async Task<byte[]> ReadAllBytesAsync(string filePath, CancellationToken cancellationToken)
    {
        await using var file = File.OpenRead(filePath);

        if (!filePath.EndsWith(".gz", StringComparison.OrdinalIgnoreCase))
        {
            using var plainBuffer = new MemoryStream();
            await file.CopyToAsync(plainBuffer, cancellationToken).ConfigureAwait(false);

            return plainBuffer.ToArray();
        }

        await using var decompressor = new GZipStream(file, CompressionMode.Decompress);
        using var buffer = new MemoryStream();
        await decompressor.CopyToAsync(buffer, cancellationToken).ConfigureAwait(false);

        return buffer.ToArray();
    }

    private static int ReadBigEndian(byte[] bytes, int offset) =>
        BinaryPrimitives.ReadInt32BigEndian(bytes.AsSpan(offset, sizeof(int)));

    private static void GuardHeaderLength(int actualLength, int headerLength, string filePath)
    {
        if (actualLength >= headerLength)
        {
            return;
        }

        throw new InvalidDataException(
            $"The IDX file {filePath} is truncated: expected at least {headerLength} header bytes, found {actualLength}.");
    }

    private static void GuardMagicNumber(int actual, int expected, string filePath)
    {
        if (actual == expected)
        {
            return;
        }

        throw new InvalidDataException(
            $"The IDX file {filePath} has magic number {actual}, but {expected} was expected.");
    }

    private static void GuardPayloadLength(IdxPayload payload, int expectedLength, string filePath)
    {
        if (payload.Body.Length == expectedLength)
        {
            return;
        }

        throw new InvalidDataException(
            $"The IDX file {filePath} declares {expectedLength} payload bytes but carries {payload.Body.Length}.");
    }

    private sealed record IdxPayload(int[] Dimensions, byte[] Body);
}
