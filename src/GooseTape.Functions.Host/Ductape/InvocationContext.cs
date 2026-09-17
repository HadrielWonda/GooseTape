namespace GooseTape.Functions.Host.Ductape;

/// <summary>
/// The execution context Ductape attaches to an invocation, used here for correlated logging.
/// </summary>
/// <remarks>
/// Only the fields this host actually uses are modelled. Unmapped fields in the payload are
/// ignored rather than rejected, so a future SDK release that adds context can still call us.
/// </remarks>
/// <param name="InvocationId">Unique identifier for this single function call.</param>
/// <param name="Product">The Ductape product the calling feature belongs to.</param>
/// <param name="Env">The environment slug the feature is running in.</param>
/// <param name="FeatureTag">The tag of the calling feature.</param>
/// <param name="FeatureRunId">The identifier of this particular feature execution, when present.</param>
/// <param name="StepTag">The tag of the feature step that made the call.</param>
/// <param name="TraceId">The distributed trace identifier, when present.</param>
public sealed record InvocationContext(
    string InvocationId,
    string? Product,
    string? Env,
    string? FeatureTag,
    string? FeatureRunId,
    string? StepTag,
    string? TraceId);
