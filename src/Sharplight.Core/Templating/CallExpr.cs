namespace Sharplight.Templating;

internal sealed class CallExpr(Expr callee, List<Expr> args, List<(string Name, Expr Value)> named) : Expr
{
    /// <summary>The name of a <c>**mapping</c> argument in <c>named</c>: the mapping's entries are passed as keyword arguments.</summary>
    public const string Unpack = "**";

    protected override object? Core(Frame f) => Run(f, null);

    public object? EvalWithCaller(Frame f, JinjaCallable caller)
    {
        try
        {
            return Run(f, caller);
        }
        catch (TemplateException e) when (e.Annotate(Line, Col))
        {
            throw;
        }
    }

    private object? Run(Frame f, JinjaCallable? caller)
    {
        object? fn = callee is GetAttrExpr ga ? ga.Resolve(f, true) : callee.Eval(f);
        var pos = new List<object?>(args.Count);
        foreach (var a in args)
        {
            pos.Add(a.Eval(f));
        }

        Dictionary<string, object?>? nm = null;
        if (named.Count > 0)
        {
            nm = new Dictionary<string, object?>();
            foreach (var (n, v) in named)
            {
                if (n != Unpack)
                {
                    nm[n] = v.Eval(f);
                    continue;
                }

                if (v.Eval(f) is not JinjaDict mapping)
                {
                    throw new TemplateRuntimeException($"argument after ** must be a mapping ({callee.Describe()})");
                }

                foreach (var (key, value) in mapping)
                {
                    nm[key as string ?? throw new TemplateRuntimeException("keywords after ** must be strings")] = value;
                }
            }
        }
        if (caller != null)
        {
            (nm ??= new Dictionary<string, object?>())["caller"] = caller;
        }

        if (fn is JinjaCallable c)
        {
            return c.Invoke(new CallArgs(pos, nm));
        }

        if (fn is Undefined u)
        {
            throw new TemplateRuntimeException($"'{u.Name ?? callee.Describe()}' is undefined");
        }

        throw new TemplateRuntimeException($"'{callee.Describe()}' is not callable (type {Ops.TypeName(fn)})");
    }

    public override string Describe() => callee.Describe() + "(...)";
}
