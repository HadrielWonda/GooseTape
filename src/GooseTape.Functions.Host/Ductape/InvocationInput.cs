using System.Text.Json;

namespace GooseTape.Functions.Host.Ductape;

/// <summary>
/// Reads and validates the fields of a portable function input payload.
/// </summary>
/// <remarks>
/// Ductape validates input against the JSON schema declared in the contract before dispatching,
/// but this host is reachable by anything holding the access key, so every field is re-checked
/// here. Failures surface as <c>FUNCTION_INPUT_INVALID</c> naming the offending field, rather
/// than as an unhandled cast deep inside an operation.
/// </remarks>
public sealed class InvocationInput
{
    private const string InvalidInputCode = "FUNCTION_INPUT_INVALID";

    private readonly JsonElement _payload;

    private InvocationInput(JsonElement payload) => _payload = payload;

    /// <summary>
    /// Wraps a raw input payload.
    /// </summary>
    /// <param name="payload">The raw input element from the invocation body.</param>
    /// <returns>A new <see cref="InvocationInput"/>.</returns>
    /// <exception cref="PortableFunctionException">Thrown when the payload is not a JSON object.</exception>
    public static InvocationInput From(JsonElement payload)
    {
        if (payload.ValueKind != JsonValueKind.Object)
        {
            throw new PortableFunctionException(
                InvalidInputCode,
                $"Function input must be a JSON object, but was {payload.ValueKind}.");
        }

        return new InvocationInput(payload);
    }

    /// <summary>Reads a required, non-blank string field.</summary>
    /// <param name="field">The field name.</param>
    /// <returns>The field value.</returns>
    /// <exception cref="PortableFunctionException">Thrown when the field is missing, not a string, or blank.</exception>
    public string RequireString(string field)
    {
        var element = Require(field, JsonValueKind.String);
        var value = element.GetString();

        if (string.IsNullOrWhiteSpace(value))
        {
            throw Invalid(field, "must not be blank");
        }

        return value;
    }

    /// <summary>Reads a required integer field, bounded inclusively.</summary>
    /// <param name="field">The field name.</param>
    /// <param name="smallest">The smallest permitted value.</param>
    /// <param name="largest">The largest permitted value.</param>
    /// <returns>The field value.</returns>
    /// <exception cref="PortableFunctionException">Thrown when the field is missing, not an integer, or out of range.</exception>
    public int RequireInteger(string field, int smallest, int largest)
    {
        var element = Require(field, JsonValueKind.Number);

        if (!element.TryGetInt32(out var value))
        {
            throw Invalid(field, "must be a 32 bit integer");
        }

        if (value < smallest || value > largest)
        {
            throw Invalid(field, $"must lie within [{smallest}, {largest}] but was {value}");
        }

        return value;
    }

    /// <summary>Reads an optional integer field, bounded inclusively.</summary>
    /// <param name="field">The field name.</param>
    /// <param name="smallest">The smallest permitted value.</param>
    /// <param name="largest">The largest permitted value.</param>
    /// <returns>The field value, or null when absent.</returns>
    /// <exception cref="PortableFunctionException">Thrown when the field is present but invalid.</exception>
    public int? OptionalInteger(string field, int smallest, int largest)
    {
        if (!_payload.TryGetProperty(field, out var element) || element.ValueKind == JsonValueKind.Null)
        {
            return null;
        }

        return RequireInteger(field, smallest, largest);
    }

    /// <summary>Reads a required finite number field, bounded inclusively.</summary>
    /// <param name="field">The field name.</param>
    /// <param name="smallest">The smallest permitted value.</param>
    /// <param name="largest">The largest permitted value.</param>
    /// <returns>The field value.</returns>
    /// <exception cref="PortableFunctionException">Thrown when the field is missing, not finite, or out of range.</exception>
    public double RequireNumber(string field, double smallest, double largest)
    {
        var element = Require(field, JsonValueKind.Number);

        if (!element.TryGetDouble(out var value) || !double.IsFinite(value))
        {
            throw Invalid(field, "must be a finite number");
        }

        if (value < smallest || value > largest)
        {
            throw Invalid(field, $"must lie within [{smallest}, {largest}] but was {value}");
        }

        return value;
    }

    /// <summary>Reads a required array of finite numbers.</summary>
    /// <param name="field">The field name.</param>
    /// <param name="fewest">The smallest permitted element count.</param>
    /// <param name="most">The largest permitted element count.</param>
    /// <returns>The field values.</returns>
    /// <exception cref="PortableFunctionException">
    /// Thrown when the field is missing, is not an array, has the wrong length, or holds a
    /// value that is not a finite number.
    /// </exception>
    public IReadOnlyList<double> RequireNumberArray(string field, int fewest, int most)
    {
        var element = Require(field, JsonValueKind.Array);
        var length = element.GetArrayLength();

        if (length < fewest || length > most)
        {
            throw Invalid(field, $"must hold between {fewest} and {most} values but held {length}");
        }

        var values = new double[length];
        var index = 0;

        foreach (var item in element.EnumerateArray())
        {
            values[index] = ReadNumberAt(field, item, index);
            index++;
        }

        return values;
    }

    /// <summary>Reads a required array of integers.</summary>
    /// <param name="field">The field name.</param>
    /// <param name="fewest">The smallest permitted element count.</param>
    /// <param name="most">The largest permitted element count.</param>
    /// <returns>The field values.</returns>
    /// <exception cref="PortableFunctionException">
    /// Thrown when the field is missing, is not an array, has the wrong length, or holds a
    /// value that is not an integer.
    /// </exception>
    public IReadOnlyList<int> RequireIntegerArray(string field, int fewest, int most)
    {
        var element = Require(field, JsonValueKind.Array);
        var length = element.GetArrayLength();

        if (length < fewest || length > most)
        {
            throw Invalid(field, $"must hold between {fewest} and {most} values but held {length}");
        }

        var values = new int[length];
        var index = 0;

        foreach (var item in element.EnumerateArray())
        {
            values[index] = ReadIntegerAt(field, item, index);
            index++;
        }

        return values;
    }

    private static double ReadNumberAt(string field, JsonElement item, int index)
    {
        if (item.ValueKind != JsonValueKind.Number || !item.TryGetDouble(out var value) || !double.IsFinite(value))
        {
            throw Invalid(field, $"must hold only finite numbers, but index {index} did not");
        }

        return value;
    }

    private static int ReadIntegerAt(string field, JsonElement item, int index)
    {
        if (item.ValueKind != JsonValueKind.Number || !item.TryGetInt32(out var value))
        {
            throw Invalid(field, $"must hold only 32 bit integers, but index {index} did not");
        }

        return value;
    }

    private JsonElement Require(string field, JsonValueKind expected)
    {
        if (!_payload.TryGetProperty(field, out var element))
        {
            throw Invalid(field, "is required but was not supplied");
        }

        if (element.ValueKind != expected)
        {
            throw Invalid(field, $"must be of type {expected} but was {element.ValueKind}");
        }

        return element;
    }

    private static PortableFunctionException Invalid(string field, string problem) =>
        new(InvalidInputCode, $"Function input field {field} {problem}.");
}
