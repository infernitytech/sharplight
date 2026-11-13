using System.Text;

namespace Sharplight.Templating;

internal sealed class ForStmt(Target target, Expr iter, Expr? filter, List<Stmt> body, List<Stmt>? otherwise) : Stmt
{
    public override Flow Exec(Frame f, StringBuilder o)
    {
        List<object?> seq;
        try
        {
            seq = Ops.Iterate(iter.Eval(f));
        }
        catch (TemplateException ex) when (ex.Annotate(Line, Col))
        {
            throw;
        }
        var inner = new Frame(f.Env, f);
        var items = seq;
        if (filter != null)
        {
            items = new List<object?>();
            foreach (var x in seq)
            {
                target.Assign(inner, x);
                if (Ops.Truthy(filter.Eval(inner)))
                {
                    items.Add(x);
                }
            }
        }
        if (items.Count == 0)
        {
            if (otherwise != null)
            {
                ExecAll(otherwise, f, o);
            }

            return Flow.Normal;
        }
        var loop = new LoopContext { Items = items };
        inner.Vars["loop"] = loop;
        inner.LoopDepth = f.LoopDepth + 1;
        for (int i = 0; i < items.Count; i++)
        {
            loop.Index0 = i;
            try
            {
                target.Assign(inner, items[i]);
            }
            catch (TemplateException ex) when (ex.Annotate(Line, Col))
            {
                throw;
            }
            var flow = ExecAll(body, inner, o);
            if (flow == Flow.Break)
            {
                break;
            }
        }
        return Flow.Normal;
    }
}
