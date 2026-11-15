namespace Sharplight.Templating;

/// <summary>A host function exposed to templates.</summary>
internal sealed class NativeFunction : JinjaCallable
{
    private readonly Func<CallArgs, object?> _fn;

    public NativeFunction(string name, Func<CallArgs, object?> fn)
    {
        Name = name;
        _fn = fn;
    }

    public override string Name { get; }
    public override object? Invoke(CallArgs args) => _fn(args);
}
