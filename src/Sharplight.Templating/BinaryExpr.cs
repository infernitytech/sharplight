namespace Sharplight.Templating;

internal sealed class BinaryExpr(string op, Expr left, Expr right) : Expr
{
    protected override object? Core(Frame f)
    {
        var a = left.Eval(f);
        var b = right.Eval(f);
        return op switch
        {
            "+" => Ops.Add(a, b),
            "-" => Ops.Sub(a, b),
            "*" => Ops.Mul(a, b),
            "/" => Ops.Div(a, b),
            "//" => Ops.FloorDiv(a, b),
            "%" => Ops.Mod(a, b),
            "**" => Ops.Pow(a, b),
            _ => throw new InvalidOperationException(op),
        };
    }
}
