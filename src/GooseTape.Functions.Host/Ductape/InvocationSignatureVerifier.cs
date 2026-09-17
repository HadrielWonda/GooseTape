using System.Security.Cryptography;
using System.Text;

namespace GooseTape.Functions.Host.Ductape;

/// <summary>
/// Verifies the signature Ductape attaches to every portable function invocation.
/// </summary>
/// <remarks>
/// Mirrors the SDK verifier: the signature must match, and the timestamp must fall inside a
/// five minute window so a captured request cannot be replayed indefinitely. Comparison is
/// fixed-time, because a byte-by-byte comparison that returns early leaks how much of a forged
/// signature was correct.
/// </remarks>
public sealed class InvocationSignatureVerifier
{
    private readonly DuctapeAccessKey _accessKey;
    private readonly TimeSpan _tolerance;

    private InvocationSignatureVerifier(DuctapeAccessKey accessKey, TimeSpan tolerance)
    {
        _accessKey = accessKey;
        _tolerance = tolerance;
    }

    /// <summary>The replay window the Ductape SDK applies, in milliseconds.</summary>
    public const int DefaultToleranceMilliseconds = 300_000;

    /// <summary>
    /// Creates a verifier over an access key.
    /// </summary>
    /// <param name="accessKey">The shared secret invocations are signed with.</param>
    /// <param name="tolerance">
    /// How far the request timestamp may sit from now, in either direction. Defaults to five minutes.
    /// </param>
    /// <returns>A new <see cref="InvocationSignatureVerifier"/>.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="accessKey"/> is null.</exception>
    public static InvocationSignatureVerifier Create(DuctapeAccessKey accessKey, TimeSpan? tolerance = null)
    {
        ArgumentNullException.ThrowIfNull(accessKey);

        return new InvocationSignatureVerifier(
            accessKey,
            tolerance ?? TimeSpan.FromMilliseconds(DefaultToleranceMilliseconds));
    }

    /// <summary>
    /// Determines whether a request carries a valid, unexpired signature.
    /// </summary>
    /// <param name="timestamp">The request timestamp header, as milliseconds since the Unix epoch.</param>
    /// <param name="signature">The request signature header, hex encoded.</param>
    /// <param name="body">The exact raw request body.</param>
    /// <param name="now">The current time, supplied so the window is testable.</param>
    /// <returns><see langword="true"/> when the request is authentic and within the replay window.</returns>
    public bool IsValid(string? timestamp, string? signature, string body, DateTimeOffset now)
    {
        ArgumentNullException.ThrowIfNull(body);

        if (string.IsNullOrWhiteSpace(timestamp) || string.IsNullOrWhiteSpace(signature))
        {
            return false;
        }

        if (!IsWithinWindow(timestamp, now))
        {
            return false;
        }

        return MatchesSignature(timestamp, signature, body);
    }

    private bool IsWithinWindow(string timestamp, DateTimeOffset now)
    {
        if (!long.TryParse(timestamp, out var milliseconds))
        {
            return false;
        }

        var sentAt = DateTimeOffset.FromUnixTimeMilliseconds(milliseconds);

        return (now - sentAt).Duration() <= _tolerance;
    }

    private bool MatchesSignature(string timestamp, string signature, string body)
    {
        var expected = Encoding.UTF8.GetBytes(_accessKey.ComputeSignature(timestamp, body));
        var supplied = Encoding.UTF8.GetBytes(signature);

        return expected.Length == supplied.Length && CryptographicOperations.FixedTimeEquals(expected, supplied);
    }
}
