using System.Text;

namespace Sharplight.Templating;

internal sealed class LoopControlStmt(Flow flow) : Stmt
{
    public override Flow Exec(Frame f, StringBuilder o) => flow;
}
