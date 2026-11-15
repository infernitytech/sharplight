namespace Sharplight.Templating;

/// <summary>
/// Value of a name or attribute that does not exist. Renders as empty text, is falsy and iterates as empty
/// (unless <see cref="Strict"/>, in which case any use except <c>is defined</c>-style tests throws).
/// </summary>
internal sealed class Undefined
{
    internal Undefined(string? name, bool strict)
    {
        Name = name;
        Strict = strict;
    }

    public string? Name { get; }
    public bool Strict { get; }
    public static Undefined Default { get; } = new(null, false);

    internal TemplateRuntimeException Fail() => new($"'{Name ?? "value"}' is undefined");

    public override string ToString() => "";
}
