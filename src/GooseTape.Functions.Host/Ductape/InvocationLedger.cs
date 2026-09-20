using System.Collections.Concurrent;

namespace GooseTape.Functions.Host.Ductape;

/// <summary>
/// Runs each invocation identifier at most once, and replays the recorded response afterwards.
/// </summary>
/// <remarks>
/// <para>
/// Signature verification with a timestamp window bounds how long a captured request stays
/// valid, but within that window the same request can be sent repeatedly. Ductape mints a fresh
/// <c>invocation_id</c> for every attempt it makes, so a repeat of an identifier is a replay
/// rather than a legitimate retry, and is answered from this ledger instead of being executed
/// again.
/// </para>
/// <para>
/// Entries are held for the same window the signature verifier accepts. Once a request is too
/// old to verify it can no longer be replayed, so remembering it longer would only consume
/// memory. Recording is per process: it makes one host's execution single-use, and is not a
/// distributed idempotency store.
/// </para>
/// </remarks>
public sealed class InvocationLedger
{
    /// <summary>The entry count above which expired entries are swept before adding another.</summary>
    private const int SweepThreshold = 512;

    private readonly ConcurrentDictionary<string, LedgerEntry> _entries = new(StringComparer.Ordinal);
    private readonly LedgerWindow _window;

    private InvocationLedger(LedgerWindow window) => _window = window;

    /// <summary>
    /// Creates a ledger that remembers invocations for as long as their signatures stay valid.
    /// </summary>
    /// <param name="clock">Supplies the current time, so expiry is testable.</param>
    /// <param name="retention">How long a recorded invocation is replayable. Defaults to five minutes.</param>
    /// <returns>A new <see cref="InvocationLedger"/>.</returns>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="clock"/> is null.</exception>
    public static InvocationLedger Create(TimeProvider clock, TimeSpan? retention = null)
    {
        ArgumentNullException.ThrowIfNull(clock);

        return new InvocationLedger(new LedgerWindow(
            clock,
            retention ?? TimeSpan.FromMilliseconds(InvocationSignatureVerifier.DefaultToleranceMilliseconds)));
    }

    /// <summary>Gets the number of invocations currently remembered, for diagnostics and tests.</summary>
    public int Count => _entries.Count;

    /// <summary>
    /// Executes an invocation, or returns the response an earlier attempt with the same
    /// identifier produced.
    /// </summary>
    /// <param name="invocationId">The invocation identifier taken from the signed body.</param>
    /// <param name="execute">Runs the invocation. Called at most once per identifier.</param>
    /// <returns>The response for this invocation.</returns>
    /// <exception cref="ArgumentException">Thrown when <paramref name="invocationId"/> is null or blank.</exception>
    /// <exception cref="ArgumentNullException">Thrown when <paramref name="execute"/> is null.</exception>
    public async Task<RecordedInvocation> ExecuteOnceAsync(
        string invocationId,
        Func<Task<RecordedInvocation>> execute)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(invocationId);
        ArgumentNullException.ThrowIfNull(execute);

        SweepIfCrowded();

        // GetOrAdd can build more than one Lazy under contention, but only the stored one is ever
        // returned, and Lazy runs its factory once, so the invocation still executes exactly once.
        var entry = _entries.GetOrAdd(
            invocationId,
            _ => new LedgerEntry(
                new Lazy<Task<RecordedInvocation>>(execute, LazyThreadSafetyMode.ExecutionAndPublication),
                _window.ExpiryFromNow()));

        try
        {
            return await entry.Response.Value.ConfigureAwait(false);
        }
        catch
        {
            // A throwing invocation is not a recorded outcome; let the next attempt run again.
            _entries.TryRemove(invocationId, out _);
            throw;
        }
    }

    /// <summary>Reports whether an invocation identifier has already been recorded.</summary>
    /// <param name="invocationId">The identifier to test.</param>
    /// <returns><see langword="true"/> when a response is being held for it.</returns>
    public bool Contains(string invocationId) =>
        !string.IsNullOrWhiteSpace(invocationId)
        && _entries.TryGetValue(invocationId, out var entry)
        && !_window.HasExpired(entry);

    private void SweepIfCrowded()
    {
        if (_entries.Count < SweepThreshold)
        {
            return;
        }

        foreach (var pair in _entries)
        {
            if (_window.HasExpired(pair.Value))
            {
                _entries.TryRemove(pair.Key, out _);
            }
        }
    }

    private sealed record LedgerEntry(Lazy<Task<RecordedInvocation>> Response, DateTimeOffset ExpiresAt);

    private sealed record LedgerWindow(TimeProvider Clock, TimeSpan Retention)
    {
        public DateTimeOffset ExpiryFromNow() => Clock.GetUtcNow() + Retention;

        public bool HasExpired(LedgerEntry entry) => Clock.GetUtcNow() >= entry.ExpiresAt;
    }
}
