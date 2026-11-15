using System.Text;

namespace Sharplight.Templating;

internal sealed class FilterBlockStmt(List<FilterSpec> filters, List<Stmt> body) : Stmt
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

        o.Append(Ops.Str(v));
        return Flow.Normal;
    }
}
