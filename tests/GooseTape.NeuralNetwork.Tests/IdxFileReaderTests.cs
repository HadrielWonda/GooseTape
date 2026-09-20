using System.Buffers.Binary;
using System.IO.Compression;
using GooseTape.NeuralNetwork.Data;

namespace GooseTape.NeuralNetwork.Tests;

/// <summary>
/// Covers the MNIST IDX reader against files built in memory, including the malformed cases a
/// truncated or mismatched download would produce.
/// </summary>
public sealed class IdxFileReaderTests : IDisposable
{
    private const int ImageMagicNumber = 2051;
    private const int LabelMagicNumber = 2049;

    private readonly string _directory =
        Path.Combine(Path.GetTempPath(), $"goosetape-idx-tests-{Guid.NewGuid():N}");

    public IdxFileReaderTests() => Directory.CreateDirectory(_directory);

    [Fact]
    public async Task ReadImages_ParsesEveryImageAndNormalisesPixels()
    {
        // Two 2x2 images: the first spans the full greyscale range, the second is uniform.
        var path = WriteFile("images.idx", IdxBytes(ImageMagicNumber, [2, 2, 2], [0, 255, 51, 204, 128, 128, 128, 128]));

        var images = await IdxFileReader.ReadImagesAsync(path);

        images.Should().HaveCount(2);
        images[0].Count.Should().Be(4);
        images[0].Intensities[0].Should().Be(0d);
        images[0].Intensities[1].Should().Be(1d);
        images[0].Intensities[2].Should().BeApproximately(0.2d, 1e-9);
        images[1].Intensities.ToArray().Should().AllSatisfy(value => value.Should().BeApproximately(128d / 255d, 1e-9));
    }

    [Fact]
    public async Task ReadLabels_ParsesEveryLabel()
    {
        var path = WriteFile("labels.idx", IdxBytes(LabelMagicNumber, [4], [0, 9, 4, 7]));

        var labels = await IdxFileReader.ReadLabelsAsync(path);

        labels.Select(label => label.Value).Should().Equal(0, 9, 4, 7);
    }

    [Fact]
    public async Task GzippedFiles_AreReadTransparently()
    {
        // The canonical MNIST download ships gzipped, so this is the normal path in practice.
        var path = WriteGzippedFile("labels.idx.gz", IdxBytes(LabelMagicNumber, [3], [1, 2, 3]));

        var labels = await IdxFileReader.ReadLabelsAsync(path);

        labels.Select(label => label.Value).Should().Equal(1, 2, 3);
    }

    [Fact]
    public async Task ImageFileWithLabelMagicNumber_IsRejected()
    {
        var path = WriteFile("wrong-magic.idx", IdxBytes(LabelMagicNumber, [1, 1, 1], [7]));

        var act = async () => await IdxFileReader.ReadImagesAsync(path);

        (await act.Should().ThrowAsync<InvalidDataException>())
            .WithMessage($"*{LabelMagicNumber}*{ImageMagicNumber}*");
    }

    [Fact]
    public async Task TruncatedHeader_IsRejected()
    {
        var path = WriteFile("truncated.idx", [0, 0, 8, 3, 0, 0]);

        var act = async () => await IdxFileReader.ReadImagesAsync(path);

        (await act.Should().ThrowAsync<InvalidDataException>()).WithMessage("*truncated*");
    }

    [Fact]
    public async Task PayloadShorterThanTheHeaderDeclares_IsRejected()
    {
        // Declares two 2x2 images, but carries only one image worth of pixels.
        var path = WriteFile("short.idx", IdxBytes(ImageMagicNumber, [2, 2, 2], [1, 2, 3, 4]));

        var act = async () => await IdxFileReader.ReadImagesAsync(path);

        (await act.Should().ThrowAsync<InvalidDataException>()).WithMessage("*declares 8 payload bytes*carries 4*");
    }

    [Fact]
    public async Task LabelOutsideTheDigitRange_IsRejected()
    {
        var path = WriteFile("bad-label.idx", IdxBytes(LabelMagicNumber, [2], [3, 17]));

        var act = async () => await IdxFileReader.ReadLabelsAsync(path);

        await act.Should().ThrowAsync<ArgumentOutOfRangeException>();
    }

    [Fact]
    public async Task MissingFile_IsReportedAsFileNotFound()
    {
        var act = async () => await IdxFileReader.ReadLabelsAsync(Path.Combine(_directory, "absent.idx"));

        await act.Should().ThrowAsync<FileNotFoundException>();
    }

    [Theory]
    [InlineData("")]
    [InlineData("   ")]
    public async Task BlankPath_IsRejected(string path)
    {
        var act = async () => await IdxFileReader.ReadImagesAsync(path);

        await act.Should().ThrowAsync<ArgumentException>();
    }

    [Fact]
    public async Task DatasetSource_PairsImagesWithTheirLabels()
    {
        var images = WriteFile("pair-images.idx", IdxBytes(ImageMagicNumber, [3, 1, 2], [255, 0, 0, 255, 128, 128]));
        var labels = WriteFile("pair-labels.idx", IdxBytes(LabelMagicNumber, [3], [8, 1, 5]));

        var dataset = await IdxDatasetSource.Create(images, labels).LoadAsync();

        dataset.Count.Should().Be(3);
        dataset.PixelCount.Should().Be(2);
        dataset.Select(image => image.Label.Value).Should().Equal(8, 1, 5);
        dataset[0].Pixels.Intensities.ToArray().Should().Equal(1d, 0d);
    }

    [Fact]
    public async Task DatasetSource_WithMismatchedCounts_IsRejected()
    {
        var images = WriteFile("count-images.idx", IdxBytes(ImageMagicNumber, [2, 1, 1], [1, 2]));
        var labels = WriteFile("count-labels.idx", IdxBytes(LabelMagicNumber, [3], [1, 2, 3]));

        var act = async () => await IdxDatasetSource.Create(images, labels).LoadAsync();

        (await act.Should().ThrowAsync<InvalidDataException>()).WithMessage("*2 images*3 labels*");
    }

    [Theory]
    [InlineData("", "labels.idx")]
    [InlineData("images.idx", "")]
    public void DatasetSource_WithABlankPath_IsRejected(string imagePath, string labelPath)
    {
        var act = () => IdxDatasetSource.Create(imagePath, labelPath);

        act.Should().Throw<ArgumentException>();
    }

    /// <summary>Builds an IDX file: a big-endian magic number, one dimension per axis, then the payload.</summary>
    private static byte[] IdxBytes(int magicNumber, int[] dimensions, byte[] payload)
    {
        var bytes = new byte[sizeof(int) * (1 + dimensions.Length) + payload.Length];
        BinaryPrimitives.WriteInt32BigEndian(bytes, magicNumber);

        for (var axis = 0; axis < dimensions.Length; axis++)
        {
            BinaryPrimitives.WriteInt32BigEndian(bytes.AsSpan(sizeof(int) * (axis + 1)), dimensions[axis]);
        }

        payload.CopyTo(bytes.AsSpan(sizeof(int) * (1 + dimensions.Length)));

        return bytes;
    }

    private string WriteFile(string name, byte[] contents)
    {
        var path = Path.Combine(_directory, name);
        File.WriteAllBytes(path, contents);

        return path;
    }

    private string WriteGzippedFile(string name, byte[] contents)
    {
        var path = Path.Combine(_directory, name);

        using var file = File.Create(path);
        using var compressor = new GZipStream(file, CompressionLevel.Optimal);
        compressor.Write(contents);

        return path;
    }

    public void Dispose()
    {
        if (Directory.Exists(_directory))
        {
            Directory.Delete(_directory, recursive: true);
        }
    }
}
