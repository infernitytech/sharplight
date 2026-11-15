namespace Sharplight.Templating;

internal sealed class OrExpr(Expr l, Expr r) : Expr
{
    protected override object? Core(Frame f)
    {
        var a = l.Eval(f);
        return Ops.Truthy(a) ? a : r.Eval(f);
    }
}
