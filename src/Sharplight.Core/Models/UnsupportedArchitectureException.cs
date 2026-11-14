namespace Sharplight.Models;

/// <summary>An importer has no mapping for the model's architecture.</summary>
public sealed class UnsupportedArchitectureException : NotSupportedException
{
    /// <summary>Creates the exception.</summary>
    /// <param name="architecture">
    /// The architecture name as the file states it (e.g. a GGUF <c>general.architecture</c> or a Hugging Face <c>model_type</c>).
    /// </param>
    /// <param name="message">The full message.</param>
    public UnsupportedArchitectureException(string architecture, string message)
        : base(message)
    {
        Architecture = architecture;
    }

    /// <summary>The architecture name as the file states it.</summary>
    public string Architecture { get; }
}
