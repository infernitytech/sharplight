namespace Sharplight.Templating;

internal sealed class NegExpr(Expr e, bool neg) : Expr
{
    protected override object? Core(Frame f) => neg ? Ops.Neg(e.Eval(f)) : Ops.Pos(e.Eval(f));
}
