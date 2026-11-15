using System.Text;

namespace Sharplight.Templating;

internal sealed class PrintStmt(Expr e) : Stmt
{
    public override Flow Exec(Frame f, StringBuilder o)
    {
        var v = e.Eval(f);
        try
        {
            o.Append(Ops.Str(v));
        }
        catch (TemplateException ex) when (ex.Annotate(Line, Col))
        {
            throw;
        }
        return Flow.Normal;
    }
}
