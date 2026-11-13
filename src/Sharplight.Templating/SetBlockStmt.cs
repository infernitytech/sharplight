using System.Text;

namespace Sharplight.Templating;

internal sealed class SetBlockStmt(Target target, List<FilterSpec> filters, List<Stmt> body) : Stmt
{
    public override Flow Exec(Frame f, StringBuilder o)
    {
        var sb = new StringBuilder();
        ExecAll(body, f, sb);
        object? v = sb.ToString();
        foreach (var fl in filters)
        {
            v = fl.Apply(f, v);
        }

        target.Assign(f, v);
        return Flow.Normal;
    }
}
