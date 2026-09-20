namespace GooseTape.NeuralNetwork.Data;

/// <summary>
/// Supplies the labelled images a training or evaluation run consumes.
/// </summary>
/// <remarks>
/// Kept as an abstraction so the function host can be pointed at the real MNIST files in
/// production and at a deterministic in-memory set under test, without either knowing about
/// the other.
/// </remarks>
public interface IImageDatasetSource
{
    /// <summary>
    /// Loads the dataset.
    /// </summary>
    /// <param name="cancellationToken">Token to monitor for cancellation requests.</param>
    /// <returns>The loaded dataset.</returns>
    /// <exception cref="InvalidDataException">Thrown when the underlying data is malformed.</exception>
    Task<ImageDataset> LoadAsync(CancellationToken cancellationToken = default);
}
