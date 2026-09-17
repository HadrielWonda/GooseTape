namespace GooseTape.Functions.Host.Ductape;

/// <summary>
/// Raised when an operation cannot complete, carrying the code Ductape should see.
/// </summary>
public sealed class PortableFunctionException : Exception
{
    /// <summary>Initialises a new instance.</summary>
    /// <param name="code">A stable machine readable failure code.</param>
    /// <param name="message">A human readable description, free of secrets.</param>
    /// <param name="retryable">Whether Ductape may retry the invocation unchanged.</param>
    public PortableFunctionException(string code, string message, bool retryable = false)
        : base(message)
    {
        Code = code;
        Retryable = retryable;
    }

    /// <summary>Initialises a new instance with a message.</summary>
    /// <param name="message">A human readable description.</param>
    public PortableFunctionException(string message)
        : base(message)
    {
        Code = "FUNCTION_EXECUTION_FAILED";
        Retryable = false;
    }

    /// <summary>Initialises a new instance with a message and inner exception.</summary>
    /// <param name="message">A human readable description.</param>
    /// <param name="innerException">The underlying cause.</param>
    public PortableFunctionException(string message, Exception innerException)
        : base(message, innerException)
    {
        Code = "FUNCTION_EXECUTION_FAILED";
        Retryable = false;
    }

    /// <summary>Initialises a new instance.</summary>
    public PortableFunctionException()
        : this("The portable function operation failed.")
    {
    }

    /// <summary>Gets the machine readable failure code.</summary>
    public string Code { get; }

    /// <summary>Gets whether Ductape may retry the invocation unchanged.</summary>
    public bool Retryable { get; }
}
