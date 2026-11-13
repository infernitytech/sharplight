namespace Sharplight.Templating;

internal sealed class NameExpr(string name) : Expr
{
    public string Name => name;
    protected override object? Core(Frame f) => f.Lookup(name);
    public override string Describe() => name;
}
