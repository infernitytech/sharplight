using System.Text;

namespace Sharplight.Templating;

internal sealed class DataStmt(string text) : Stmt
{
    public override Flow Exec(Frame f, StringBuilder o)
    {
        o.Append(text);
        return Flow.Normal;
    }
}
