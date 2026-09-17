namespace GooseTape.Functions.Host.Tests;

/// <summary>
/// Shares one in-memory host across a test class.
/// </summary>
/// <remarks>
/// The host loads and caches its dataset on first use, so rebuilding it per test would pay that
/// cost repeatedly for no added coverage.
/// </remarks>
public sealed class FunctionHostFactoryFixture : IDisposable
{
    internal FunctionHostFactory Factory { get; } = new();

    public void Dispose() => Factory.Dispose();
}
