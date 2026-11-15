namespace Sharplight.Templating;

/// <summary>Thrown for lexing and parsing errors.</summary>
public sealed class TemplateSyntaxException : TemplateException
{
    /// <summary>Creates an exception with an optional 1-based source position (0 = unknown).</summary>
    public TemplateSyntaxException(string message, int line = 0, int column = 0) : base(message, line, column) { }
}
