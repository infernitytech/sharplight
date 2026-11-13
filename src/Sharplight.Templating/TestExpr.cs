namespace Sharplight.Templating;

internal sealed class TestExpr(Expr value, string name, List<Expr> args, List<(string Name, Expr Value)> named, bool negate) : Expr
{
    protected override object? Core(Frame f)
    {
        if (!f.Env.Tests.TryGetValue(name, out var fn))
        {
            throw new TemplateRuntimeException($"No test named '{name}'");
        }

        var v = value.Eval(f);
        var pos = new List<object?>(args.Count);
        foreach (var a in args)
        {
            pos.Add(a.Eval(f));
        }

        Dictionary<string, object?>? nm = null;
        if (named.Count > 0)
        {
            nm = new Dictionary<string, object?>();
            foreach (var (n, nv) in named)
            {
                nm[n] = nv.Eval(f);
            }
        }
        bool r = fn(f.Env, v, new CallArgs(pos, nm));
        return negate ? !r : r;
    }
}
