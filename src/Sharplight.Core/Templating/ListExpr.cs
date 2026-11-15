namespace Sharplight.Templating;

internal sealed class ListExpr(List<Expr> items, bool tuple = false) : Expr
{
    protected override object? Core(Frame f)
    {
        List<object?> l = tuple ? new JinjaTuple() : new List<object?>(items.Count);
        foreach (var e in items)
        {
            l.Add(e.Eval(f));
        }

        return l;
    }
}
