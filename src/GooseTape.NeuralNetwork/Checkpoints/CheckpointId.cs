using System.Text.RegularExpressions;

namespace GooseTape.NeuralNetwork.Checkpoints;

/// <summary>
/// The identifier of a persisted network snapshot.
/// </summary>
/// <remarks>
/// The permitted character set is deliberately narrow. Checkpoint identifiers arrive from
/// outside the process as portable function input and are used to address stored files, so
/// anything resembling a path separator or a relative segment must be rejected at the boundary
/// rather than sanitised further in.
/// </remarks>
/// <param name="Value">The identifier.</param>
public readonly partial record struct CheckpointId(string Value)
{
    private const int LongestIdentifier = 128;

    /// <summary>
    /// Creates a validated checkpoint identifier.
    /// </summary>
    /// <param name="value">
    /// The identifier. Must be 1 to 128 characters of ASCII letters, digits, hyphen or underscore.
    /// </param>
    /// <returns>A validated <see cref="CheckpointId"/>.</returns>
    /// <exception cref="ArgumentException">
    /// Thrown when <paramref name="value"/> is blank, over-long, or holds an unsupported character.
    /// </exception>
    public static CheckpointId Create(string value)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(value);

        if (value.Length > LongestIdentifier)
        {
            throw new ArgumentException(
                $"A checkpoint identifier may be at most {LongestIdentifier} characters. Received {value.Length}.",
                nameof(value));
        }

        if (!SupportedCharacters().IsMatch(value))
        {
            throw new ArgumentException(
                "A checkpoint identifier may only hold ASCII letters, digits, hyphens and underscores.",
                nameof(value));
        }

        return new CheckpointId(value);
    }

    /// <summary>Creates an identifier for a given run and epoch.</summary>
    /// <param name="runIdentifier">The identifier of the training run.</param>
    /// <param name="epoch">The one based epoch the checkpoint was taken after.</param>
    /// <returns>A validated <see cref="CheckpointId"/>.</returns>
    /// <exception cref="ArgumentException">Thrown when the composed identifier is not valid.</exception>
    public static CheckpointId ForEpoch(string runIdentifier, int epoch) =>
        Create($"{runIdentifier}-epoch-{epoch}");

    /// <summary>Returns the identifier.</summary>
    /// <returns>The underlying string.</returns>
    public override string ToString() => Value;

    [GeneratedRegex(@"\A[A-Za-z0-9_-]+\z", RegexOptions.CultureInvariant)]
    private static partial Regex SupportedCharacters();
}
