using System.Text.Json;

namespace GooseTape.NeuralNetwork.Checkpoints;

/// <summary>
/// Stores network snapshots as JSON files in a single directory.
/// </summary>
/// <remarks>
/// Writes go to a temporary file that is then moved into place. A move within one volume is
/// atomic, so a process that dies mid-write leaves the previous checkpoint intact rather than a
/// half-written file that would fail to load on resume.
/// </remarks>
public sealed class FileSystemCheckpointStore : ICheckpointStore
{
    private const string FileExtension = ".checkpoint.json";

    private readonly string _directoryPath;
    private readonly JsonSerializerOptions _serializerOptions;

    private FileSystemCheckpointStore(string directoryPath, JsonSerializerOptions serializerOptions)
    {
        _directoryPath = directoryPath;
        _serializerOptions = serializerOptions;
    }

    /// <summary>
    /// Creates a store rooted at <paramref name="directoryPath"/>, creating the directory if needed.
    /// </summary>
    /// <param name="directoryPath">The directory snapshots are written into.</param>
    /// <returns>A new <see cref="FileSystemCheckpointStore"/>.</returns>
    /// <exception cref="ArgumentException">Thrown when <paramref name="directoryPath"/> is null or blank.</exception>
    public static FileSystemCheckpointStore Create(string directoryPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directoryPath);
        Directory.CreateDirectory(directoryPath);

        return new FileSystemCheckpointStore(directoryPath, new JsonSerializerOptions { WriteIndented = false });
    }

    /// <inheritdoc />
    public async Task SaveAsync(
        CheckpointId checkpointId,
        NetworkSnapshot snapshot,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(snapshot);

        var destinationPath = PathFor(checkpointId);
        var temporaryPath = $"{destinationPath}.{Guid.NewGuid():N}.tmp";

        await WriteAsync(temporaryPath, snapshot, cancellationToken).ConfigureAwait(false);

        File.Move(temporaryPath, destinationPath, overwrite: true);
    }

    /// <inheritdoc />
    public async Task<NetworkSnapshot> LoadAsync(
        CheckpointId checkpointId,
        CancellationToken cancellationToken = default)
    {
        var path = PathFor(checkpointId);

        if (!File.Exists(path))
        {
            throw new CheckpointNotFoundException(checkpointId);
        }

        await using var file = File.OpenRead(path);

        var snapshot = await JsonSerializer
            .DeserializeAsync<NetworkSnapshot>(file, _serializerOptions, cancellationToken)
            .ConfigureAwait(false);

        return snapshot ?? throw new InvalidDataException($"The checkpoint {checkpointId} deserialised to null.");
    }

    /// <inheritdoc />
    public Task<bool> ExistsAsync(CheckpointId checkpointId, CancellationToken cancellationToken = default) =>
        Task.FromResult(File.Exists(PathFor(checkpointId)));

    private async Task WriteAsync(string path, NetworkSnapshot snapshot, CancellationToken cancellationToken)
    {
        await using var file = File.Create(path);
        await JsonSerializer.SerializeAsync(file, snapshot, _serializerOptions, cancellationToken).ConfigureAwait(false);
        await file.FlushAsync(cancellationToken).ConfigureAwait(false);
    }

    private string PathFor(CheckpointId checkpointId) =>
        Path.Combine(_directoryPath, $"{checkpointId.Value}{FileExtension}");
}
