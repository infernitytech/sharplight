namespace Sharplight.Templating;

internal sealed class FilterExpr(Expr value, FilterSpec spec) : Expr
{
    protected override object? Core(Frame f) => spec.Apply(f, value.Eval(f));
    public override string Describe() => value.Describe() + "|" + spec.Name;
}
