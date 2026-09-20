using System.Buffers.Binary;
using GooseTape.Functions.Host.Training;
using Microsoft.Extensions.Options;

namespace GooseTape.Functions.Host.Tests;

/// <summary>
/// Covers the identifier a checkpoint records to say which data it was trained on.
/// </summary>
/// <remarks>
/// Lineage checking is only as good as this identifier. If it stayed the same when the data
/// changed, a run could silently continue against a different dataset and every check would pass.
/// </remarks>
public sealed class DatasetIdentityTests : IDisposable
{
    private const int ImageMagicNumber = 2051;
    private const int LabelMagicNumber = 2049;

    private readonly string _directory =
        Path.Combine(Path.GetTempPath(), $"goosetape-dataset-id-{Guid.NewGuid():N}");

    public DatasetIdentityTests() => Directory.CreateDirectory(_directory);

    [Fact]
    public async Task TheSameFiles_AlwaysProduceTheSameIdentifier()
    {
        var options = WriteDataset("first", [10, 20, 30, 40], [1, 2]);

        var one = await new DatasetProvider(options).DatasetIdAsync();
        var two = await new DatasetProvider(options).DatasetIdAsync();

        one.Should().StartWith("idx:");
        two.Should().Be(one);
    }

    [Fact]
    public async Task ChangingThePixelsChangesTheIdentifier()
    {
        // The paths are identical; only the bytes behind them differ.
        var before = await new DatasetProvider(WriteDataset("a", [10, 20, 30, 40], [1, 2])).DatasetIdAsync();
        var after = await new DatasetProvider(WriteDataset("b", [10, 20, 30, 41], [1, 2])).DatasetIdAsync();

        after.Should().NotBe(before);
    }

    [Fact]
    public async Task ChangingTheLabelsChangesTheIdentifier()
    {
        var before = await new DatasetProvider(WriteDataset("c", [10, 20, 30, 40], [1, 2])).DatasetIdAsync();
        var after = await new DatasetProvider(WriteDataset("d", [10, 20, 30, 40], [1, 3])).DatasetIdAsync();

        after.Should().NotBe(before);
    }

    [Fact]
    public async Task GeneratedData_IsIdentifiedByWhatProducesIt()
    {
        var first = await new DatasetProvider(Synthetic(examplesPerClass: 5, seed: 99)).DatasetIdAsync();
        var same = await new DatasetProvider(Synthetic(examplesPerClass: 5, seed: 99)).DatasetIdAsync();
        var differentSeed = await new DatasetProvider(Synthetic(examplesPerClass: 5, seed: 100)).DatasetIdAsync();
        var differentSize = await new DatasetProvider(Synthetic(examplesPerClass: 6, seed: 99)).DatasetIdAsync();

        first.Should().Be(same);
        differentSeed.Should().NotBe(first);
        differentSize.Should().NotBe(first);
    }

    [Fact]
    public async Task IdxFilesAndGeneratedData_NeverShareAnIdentifier()
    {
        var idx = await new DatasetProvider(WriteDataset("e", [10, 20, 30, 40], [1, 2])).DatasetIdAsync();
        var synthetic = await new DatasetProvider(Synthetic(5, 99)).DatasetIdAsync();

        idx.Should().NotBe(synthetic);
    }

    [Fact]
    public async Task AMissingPathIsReportedRatherThanHashedAsEmpty()
    {
        var options = Options.Create(new DatasetOptions
        {
            Provider = DatasetOptions.IdxProvider,
            TrainingImagesPath = null,
            TrainingLabelsPath = null,
        });

        var act = async () => await new DatasetProvider(options).DatasetIdAsync();

        await act.Should().ThrowAsync<InvalidOperationException>().WithMessage("*TrainingImagesPath*");
    }

    private static IOptions<DatasetOptions> Synthetic(int examplesPerClass, int seed) =>
        Options.Create(new DatasetOptions
        {
            Provider = DatasetOptions.SyntheticProvider,
            SyntheticTrainingExamplesPerClass = examplesPerClass,
            SyntheticTestExamplesPerClass = examplesPerClass,
            SyntheticSeed = seed,
        });

    private IOptions<DatasetOptions> WriteDataset(string prefix, byte[] pixels, byte[] labels)
    {
        var images = Path.Combine(_directory, $"{prefix}-images.idx");
        var labelFile = Path.Combine(_directory, $"{prefix}-labels.idx");

        File.WriteAllBytes(images, Idx(ImageMagicNumber, [labels.Length, 1, pixels.Length / labels.Length], pixels));
        File.WriteAllBytes(labelFile, Idx(LabelMagicNumber, [labels.Length], labels));

        return Options.Create(new DatasetOptions
        {
            Provider = DatasetOptions.IdxProvider,
            TrainingImagesPath = images,
            TrainingLabelsPath = labelFile,
            TestImagesPath = images,
            TestLabelsPath = labelFile,
        });
    }

    private static byte[] Idx(int magicNumber, int[] dimensions, byte[] payload)
    {
        var bytes = new byte[(sizeof(int) * (1 + dimensions.Length)) + payload.Length];
        BinaryPrimitives.WriteInt32BigEndian(bytes, magicNumber);

        for (var axis = 0; axis < dimensions.Length; axis++)
        {
            BinaryPrimitives.WriteInt32BigEndian(bytes.AsSpan(sizeof(int) * (axis + 1)), dimensions[axis]);
        }

        payload.CopyTo(bytes.AsSpan(sizeof(int) * (1 + dimensions.Length)));

        return bytes;
    }

    public void Dispose()
    {
        if (Directory.Exists(_directory))
        {
            Directory.Delete(_directory, recursive: true);
        }
    }
}
