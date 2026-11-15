namespace Sharplight.Templating;

internal sealed class CondExpr(Expr cond, Expr then, Expr? otherwise) : Expr
{
    protected override object? Core(Frame f)
    {
        if (Ops.Truthy(cond.Eval(f)))
        {
            return then.Eval(f);
        }

        return otherwise != null ? otherwise.Eval(f) : new Undefined("conditional expression", f.Strict);
    }
}

// ============================================================================ assignment targets
