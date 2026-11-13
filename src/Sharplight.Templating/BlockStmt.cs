using System.Text;

namespace Sharplight.Templating;

internal sealed class BlockStmt(List<Stmt> body) : Stmt
{
    public override Flow Exec(Frame f, StringBuilder o) => ExecAll(body, f, o);
}
