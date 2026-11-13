using System.Text;

namespace Sharplight.Templating;

internal sealed class SetStmt(Target target, Expr value) : Stmt
{
    public override Flow Exec(Frame f, StringBuilder o)
    {
        var v = value.Eval(f);
        try
        {
            target.Assign(f, v);
        }
        catch (TemplateException ex) when (ex.Annotate(Line, Col))
        {
            throw;
        }
        return Flow.Normal;
    }
}
