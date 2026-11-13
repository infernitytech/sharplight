namespace Sharplight.Templating;

internal sealed class CompareExpr(Expr first, List<(string Op, Expr Rhs)> rest) : Expr
{
    protected override object? Core(Frame f)
    {
        var left = first.Eval(f);
        bool result = true;
        foreach (var (op, rhsExpr) in rest)
        {
            var right = rhsExpr.Eval(f);
            result = op switch
            {
                "==" => Ops.Eq(left, right),
                "!=" => !Ops.Eq(left, right),
                "<" => Ops.Compare(left, right, "<") < 0,
                "<=" => Ops.Compare(left, right, "<=") <= 0,
                ">" => Ops.Compare(left, right, ">") > 0,
                ">=" => Ops.Compare(left, right, ">=") >= 0,
                "in" => Ops.Contains(right, left),
                "not in" => !Ops.Contains(right, left),
                _ => throw new InvalidOperationException(op),
            };
            if (!result)
            {
                return false;
            }

            left = right;
        }
        return result;
    }
}
