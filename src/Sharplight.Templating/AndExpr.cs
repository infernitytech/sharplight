namespace Sharplight.Templating;

internal sealed class AndExpr(Expr l, Expr r) : Expr
{
    protected override object? Core(Frame f)
    {
        var a = l.Eval(f);
        return Ops.Truthy(a) ? r.Eval(f) : a;
    }
}
