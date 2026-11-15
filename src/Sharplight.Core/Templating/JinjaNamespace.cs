namespace Sharplight.Templating;

/// <summary>Result of <c>namespace(...)</c>: a mutable bag of attributes.</summary>
internal sealed class JinjaNamespace
{
    public Dictionary<string, object?> Attributes { get; } = new();
}
