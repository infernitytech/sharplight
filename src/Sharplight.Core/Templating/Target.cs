namespace Sharplight.Templating;

internal abstract class Target : Node
{
    public abstract void Assign(Frame f, object? value);
}
