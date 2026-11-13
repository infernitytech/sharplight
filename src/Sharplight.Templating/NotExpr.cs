namespace Sharplight.Templating;

internal sealed class NotExpr(Expr e) : Expr
{
    protected override object? Core(Frame f) => !Ops.Truthy(e.Eval(f));
}
