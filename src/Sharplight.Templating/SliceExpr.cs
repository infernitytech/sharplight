namespace Sharplight.Templating;

internal sealed class SliceExpr(Expr? start, Expr? stop, Expr? step) : Expr
{
    protected override object? Core(Frame f) => new SliceValue
    {
        Start = start?.Eval(f),
        Stop = stop?.Eval(f),
        Step = step?.Eval(f),
    };
}
