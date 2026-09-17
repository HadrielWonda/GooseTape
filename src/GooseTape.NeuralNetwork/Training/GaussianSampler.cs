namespace GooseTape.NeuralNetwork.Training;

/// <summary>
/// Draws normally distributed samples from a seeded uniform generator.
/// </summary>
/// <remarks>
/// Uses the Box-Muller transform. The transform yields two independent samples per pair of
/// uniforms; only the first is kept, which trades a little throughput for a generator whose
/// output depends solely on how many samples have been drawn.
/// </remarks>
public sealed class GaussianSampler
{
    private readonly Random _generator;

    private GaussianSampler(Random generator) => _generator = generator;

    /// <summary>Creates a sampler bound to <paramref name="seed"/>.</summary>
    /// <param name="seed">The seed controlling the sequence.</param>
    /// <returns>A new <see cref="GaussianSampler"/>.</returns>
    public static GaussianSampler From(RandomSeed seed) => new(seed.ToGenerator());

    /// <summary>
    /// Draws one sample from a normal distribution with mean zero.
    /// </summary>
    /// <param name="standardDeviation">The standard deviation to scale the sample by.</param>
    /// <returns>A normally distributed sample.</returns>
    public double NextSample(double standardDeviation)
    {
        var uniform = 1d - _generator.NextDouble();
        var angle = 2d * Math.PI * _generator.NextDouble();

        return Math.Sqrt(-2d * Math.Log(uniform)) * Math.Cos(angle) * standardDeviation;
    }
}
