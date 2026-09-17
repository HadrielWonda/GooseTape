using GooseTape.NeuralNetwork.Data;

namespace GooseTape.NeuralNetwork.Training;

/// <summary>
/// Runs a single training epoch: one shuffled pass over the data, one update per batch.
/// </summary>
/// <remarks>
/// An epoch is the unit of durability for this project. It takes a network in, returns a new
/// network out, and touches nothing else, so an orchestrator can checkpoint between epochs and
/// retry any single epoch without the retry observing partial state.
/// </remarks>
public sealed class EpochTrainer
{
    /// <summary>
    /// An odd multiplier used to mix the run seed with the epoch number.
    /// </summary>
    /// <remarks>
    /// Deliberately not <c>HashCode.Combine</c>. That is randomised per process, so a run
    /// replayed in a different process would shuffle differently and stop being reproducible.
    /// </remarks>
    private const int EpochSeedMultiplier = 397;

    private readonly TrainingSchedule _schedule;
    private readonly RandomSeed _seed;

    private EpochTrainer(TrainingSchedule schedule, RandomSeed seed)
    {
        _schedule = schedule;
        _seed = seed;
    }

    /// <summary>
    /// Creates a trainer bound to a schedule and a run seed.
    /// </summary>
    /// <param name="schedule">The hyperparameters to train under.</param>
    /// <param name="seed">The run seed, mixed with the epoch number to order each pass.</param>
    /// <returns>A new <see cref="EpochTrainer"/>.</returns>
    public static EpochTrainer Create(TrainingSchedule schedule, RandomSeed seed) => new(schedule, seed);

    /// <summary>
    /// Runs one epoch.
    /// </summary>
    /// <param name="network">The network to train.</param>
    /// <param name="dataset">The training data.</param>
    /// <param name="epoch">The one based epoch being run, which determines the shuffle order.</param>
    /// <returns>The network after the epoch, and the metrics observed during it.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="network"/> or <paramref name="dataset"/> is null.</exception>
    public EpochOutcome Run(FeedForwardNetwork network, ImageDataset dataset, EpochNumber epoch)
    {
        ArgumentNullException.ThrowIfNull(network);
        ArgumentNullException.ThrowIfNull(dataset);

        var accumulator = new TrainingMetricsAccumulator();
        var current = network;

        foreach (var batch in dataset.Shuffle(ShuffleSeedFor(epoch)).Batches(_schedule.BatchSize))
        {
            var outcome = current.TrainOnBatch(batch.Inputs, batch.Expected, _schedule.LearningRate);
            accumulator.Add(outcome.Metrics);
            current = outcome.Network;
        }

        return new EpochOutcome(current, accumulator.ToMetrics());
    }

    private RandomSeed ShuffleSeedFor(EpochNumber epoch) =>
        RandomSeed.Create(unchecked((_seed.Value * EpochSeedMultiplier) ^ epoch.Value));
}
