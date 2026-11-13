namespace Sharplight.Templating;

internal sealed class GetItemExpr(Expr obj, Expr key) : Expr
{
    protected override object? Core(Frame f)
    {
        var o = obj.Eval(f);
        var k = key.Eval(f);
        var r = Ops.GetItem(o, k, f.Strict);
        if (r is Undefined { Name: null } u)
        {
            return new Undefined(Describe(), u.Strict);
        }

        return r;
    }

    public override string Describe() => obj.Describe() + "[...]";
}
