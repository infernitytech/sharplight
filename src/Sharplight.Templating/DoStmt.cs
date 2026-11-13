using System.Text;

namespace Sharplight.Templating;

internal sealed class DoStmt(Expr e) : Stmt
{
    public override Flow Exec(Frame f, StringBuilder o)
    {
        e.Eval(f);
        return Flow.Normal;
    }
}
