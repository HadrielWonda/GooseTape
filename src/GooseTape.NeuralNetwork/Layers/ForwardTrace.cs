using System.Collections;
using GooseTape.NeuralNetwork.Maths;

namespace GooseTape.NeuralNetwork.Layers;

/// <summary>
/// The ordered forward passes produced by one traversal of a network, innermost layer first.
/// </summary>
public sealed class ForwardTrace : IReadOnlyList<LayerForwardPass>
{
    private readonly IReadOnlyList<LayerForwardPass> _passes;

    private ForwardTrace(IReadOnlyList<LayerForwardPass> passes) => _passes = passes;

    /// <summary>Gets the number of layers this trace covers.</summary>
    public int Count => _passes.Count;

    /// <summary>Gets the pass recorded for the layer at <paramref name="index"/>.</summary>
    /// <param name="index">The zero based layer index.</param>
    /// <returns>The forward pass for that layer.</returns>
    public LayerForwardPass this[int index] => _passes[index];

    /// <summary>Gets the activation emitted by the final layer.</summary>
    public Matrix Output => _passes[^1].Activation;

    /// <summary>
    /// Creates a trace from the passes recorded during a forward traversal.
    /// </summary>
    /// <param name="passes">The passes, ordered from the first layer to the last.</param>
    /// <returns>A new <see cref="ForwardTrace"/>.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="passes"/> is null.</exception>
    /// <exception cref="ArgumentException">Thrown when <paramref name="passes"/> is empty.</exception>
    public static ForwardTrace From(IReadOnlyList<LayerForwardPass> passes)
    {
        ArgumentNullException.ThrowIfNull(passes);

        if (passes.Count == 0)
        {
            throw new ArgumentException("A forward trace requires at least one layer pass.", nameof(passes));
        }

        return new ForwardTrace([.. passes]);
    }

    /// <inheritdoc />
    public IEnumerator<LayerForwardPass> GetEnumerator() => _passes.GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}
