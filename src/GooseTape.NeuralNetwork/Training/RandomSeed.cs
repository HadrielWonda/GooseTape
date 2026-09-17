namespace GooseTape.NeuralNetwork.Training;

/// <summary>
/// The seed that makes weight initialisation reproducible.
/// </summary>
/// <remarks>
/// Reproducibility matters here beyond convenience: a durable training run that is replayed
/// after a failure must rebuild exactly the same starting weights, or the replay is measuring
/// a different experiment.
/// </remarks>
/// <param name="Value">The seed value.</param>
public readonly record struct RandomSeed(int Value)
{
    /// <summary>Creates a seed from any integer.</summary>
    /// <param name="value">The seed value.</param>
    /// <returns>A <see cref="RandomSeed"/>.</returns>
    public static RandomSeed Create(int value) => new(value);

    /// <summary>Creates a random number generator bound to this seed.</summary>
    /// <returns>A generator that always produces the same sequence for this seed.</returns>
    public Random ToGenerator() => new(Value);
}
