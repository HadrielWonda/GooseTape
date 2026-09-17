using System.Text.Json;
using System.Text.Json.Serialization;

namespace GooseTape.Functions.Host.Ductape;

/// <summary>
/// The JSON conventions the Ductape wire protocol uses.
/// </summary>
/// <remarks>
/// Ductape spells every field in snake case, including nested context fields such as
/// <c>feature_run_id</c>. Unmapped fields are ignored so a newer SDK can still call this host.
/// </remarks>
public static class DuctapeJson
{
    /// <summary>Gets the serializer options matching the Ductape wire protocol.</summary>
    public static JsonSerializerOptions Options { get; } = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        PropertyNameCaseInsensitive = false,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        NumberHandling = JsonNumberHandling.AllowReadingFromString,
    };
}
