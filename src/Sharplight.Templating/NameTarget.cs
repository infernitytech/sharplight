namespace Sharplight.Templating;

internal sealed class NameTarget(string name) : Target
{
    public override void Assign(Frame f, object? value) => f.Set(name, value);
}
