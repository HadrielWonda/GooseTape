using System.ComponentModel.DataAnnotations;

namespace GooseTape.Functions.Host.Ductape;

/// <summary>
/// The Ductape contract this host serves, and the secret it verifies invocations with.
/// </summary>
/// <remarks>
/// The access key is never committed. It is read from the environment, from
/// <c>DUCTAPE_ACCESS_KEY</c> or from user secrets in development, and the host refuses to start
/// without it rather than serving an endpoint that would accept unsigned calls.
/// </remarks>
public sealed class DuctapeOptions
{
    /// <summary>The configuration section these options bind to.</summary>
    public const string SectionName = "Ductape";

    /// <summary>Gets or sets the Ductape SDK access key used to verify invocation signatures.</summary>
    [Required(AllowEmptyStrings = false)]
    public string AccessKey { get; set; } = string.Empty;

    /// <summary>Gets or sets the portable function namespace this host serves.</summary>
    [Required(AllowEmptyStrings = false)]
    public string FunctionNamespace { get; set; } = "goosetape.neural-network";

    /// <summary>Gets or sets the portable function contract version this host serves.</summary>
    [Required(AllowEmptyStrings = false)]
    public string FunctionVersion { get; set; } = "1";
}
