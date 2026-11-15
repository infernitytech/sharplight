namespace Sharplight.Templating;

/// <summary>Thrown for errors raised while rendering (type errors, undefined values in strict mode, ...).</summary>
public class TemplateRuntimeException : TemplateException
{
    /// <summary>Creates an exception with an optional 1-based source position (0 = unknown).</summary>
    public TemplateRuntimeException(string message, int line = 0, int column = 0) : base(message, line, column) { }
}
