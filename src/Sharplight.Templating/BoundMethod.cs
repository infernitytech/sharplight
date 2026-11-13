namespace Sharplight.Templating;

internal sealed class BoundMethod : JinjaCallable
{
    private readonly object _target;
    private readonly string _name;

    public BoundMethod(object target, string name)
    {
        _target = target;
        _name = name;
    }

    public override string Name => _name;
    public override object? Invoke(CallArgs args) => Methods.Call(_target, _name, args);
}
