using System.Text;

namespace Sharplight.Templating;

internal sealed class IfStmt(List<(Expr Cond, List<Stmt> Body)> branches, List<Stmt>? otherwise) : Stmt
{
    public override Flow Exec(Frame f, StringBuilder o)
    {
        foreach (var (cond, body) in branches)
        {
            bool t;
            try
            {
                t = Ops.Truthy(cond.Eval(f));
            }
            catch (TemplateException ex) when (ex.Annotate(Line, Col))
            {
                throw;
            }
            if (t)
            {
                return ExecAll(body, f, o);
            }
        }
        return otherwise != null ? ExecAll(otherwise, f, o) : Flow.Normal;
    }
}
