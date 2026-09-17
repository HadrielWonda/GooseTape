using System.Security.Cryptography;
using System.Text;

namespace GooseTape.Functions.Host.Ductape;

/// <summary>
/// The Ductape SDK access key, which doubles as the shared secret for function invocation signing.
/// </summary>
/// <remarks>
/// <para>
/// The key never leaves this type as a string. It is supplied through configuration, held only
/// as the bytes needed to compute a signature, and is deliberately excluded from
/// <see cref="ToString"/> so it cannot reach a log through string interpolation.
/// </para>
/// <para>
/// Ductape signs each invocation as <c>HMAC-SHA256(accessKey, timestamp + "." + body)</c>, hex
/// encoded. That scheme is mirrored here so this host can verify calls without the Node SDK.
/// </para>
/// </remarks>
public sealed class DuctapeAccessKey
{
    private readonly byte[] _keyBytes;

    private DuctapeAccessKey(byte[] keyBytes) => _keyBytes = keyBytes;

    /// <summary>
    /// Creates an access key from its configured value.
    /// </summary>
    /// <param name="value">The Ductape SDK access key.</param>
    /// <returns>A new <see cref="DuctapeAccessKey"/>.</returns>
    /// <exception cref="ArgumentException">Thrown when <paramref name="value"/> is null or blank.</exception>
    public static DuctapeAccessKey Create(string value)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(value);

        return new DuctapeAccessKey(Encoding.UTF8.GetBytes(value));
    }

    /// <summary>
    /// Computes the expected signature for a request.
    /// </summary>
    /// <param name="timestamp">The value of the request timestamp header.</param>
    /// <param name="body">The exact raw request body, byte for byte as received.</param>
    /// <returns>The lowercase hex encoded HMAC-SHA256 signature.</returns>
    /// <exception cref="ArgumentNullException">Thrown when either argument is null.</exception>
    public string ComputeSignature(string timestamp, string body)
    {
        ArgumentNullException.ThrowIfNull(timestamp);
        ArgumentNullException.ThrowIfNull(body);

        var payload = Encoding.UTF8.GetBytes($"{timestamp}.{body}");

        return Convert.ToHexStringLower(HMACSHA256.HashData(_keyBytes, payload));
    }

    /// <summary>Returns a redacted placeholder, never the key itself.</summary>
    /// <returns>A constant standing in for the key.</returns>
    public override string ToString() => "[redacted access key]";
}
