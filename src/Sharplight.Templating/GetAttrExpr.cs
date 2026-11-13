namespace Sharplight.Templating;

internal sealed class GetAttrExpr(Expr obj, string name) : Expr
{
    public Expr Object => obj;
    public string AttrName => name;

    protected override object? Core(Frame f) => Resolve(f, false);

    public object? Resolve(Frame f, bool forCall)
    {
        var o = obj.Eval(f);
        var r = Ops.GetAttr(o, name, f.Strict, forCall);
        if (r is Undefined { Name: null } u)
        {
            return new Undefined(Describe(), u.Strict);
        }

        return r;
    }

    public override string Describe() => obj.Describe() + "." + name;
}
