namespace Sharplight.Templating;

/// <summary>Something that can be called from a template: macros, builtin functions, bound methods.</summary>
internal abstract class JinjaCallable
{
    public abstract string Name { get; }
    public abstract object? Invoke(CallArgs args);
}
