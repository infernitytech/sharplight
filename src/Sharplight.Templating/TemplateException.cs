namespace Sharplight.Templating;

/// <summary>Base class of all template errors. Carries the 1-based source position when known.</summary>
public class TemplateException : Exception
{
    private readonly string _raw;

    /// <summary>Creates an exception with an optional 1-based source position (0 = unknown).</summary>
    public TemplateException(string message, int line = 0, int column = 0) : base(message)
    {
        _raw = message;
        Line = line;
        Column = column;
    }

    /// <summary>1-based line, or 0 when unknown.</summary>
    public int Line { get; private set; }

    /// <summary>1-based column, or 0 when unknown.</summary>
    public int Column { get; private set; }

    /// <summary>The message without position information.</summary>
    public string RawMessage => _raw;

    /// <summary>The message including the source position when known.</summary>
    public override string Message => Line > 0 ? $"{_raw} (line {Line}, column {Column})" : _raw;

    /// <summary>Fills in the position if it is not set yet. Always returns false (for use in exception filters).</summary>
    internal bool Annotate(int line, int column)
    {
        if (Line == 0 && line > 0)
        {
            Line = line;
            Column = column;
        }
        return false;
    }
}
