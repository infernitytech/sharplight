using System.Text;

namespace Sharplight.Templating;

internal sealed class ConcatExpr(List<Expr> parts) : Expr
{
    protected override object? Core(Frame f)
    {
        if (parts.Count == 2)
        {
            return string.Concat(Ops.Str(parts[0].Eval(f)), Ops.Str(parts[1].Eval(f)));
        }

        var sb = new StringBuilder();
        foreach (var p in parts)
        {
            sb.Append(Ops.Str(p.Eval(f)));
        }

        return sb.ToString();
    }
}
