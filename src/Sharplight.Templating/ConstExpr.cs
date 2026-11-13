namespace Sharplight.Templating;

internal sealed class ConstExpr(object? value) : Expr
{
    protected override object? Core(Frame f) => value;
    public override string Describe() => Ops.Repr(value);
}
