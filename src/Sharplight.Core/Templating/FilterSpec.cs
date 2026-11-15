namespace Sharplight.Templating;

internal sealed class FilterSpec(string name, List<Expr> args, List<(string Name, Expr Value)> named, Token tok)
{
    public string Name => name;
    public int Line => tok.Line;
    public int Col => tok.Col;

    public object? Apply(Frame f, object? value)
    {
        if (!f.Env.Filters.TryGetValue(name, out var fn))
        {
            throw new TemplateRuntimeException($"No filter named '{name}'");
        }

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
                nm[n] = v.Eval(f);
            }
        }
        try
        {
            return fn(f.Env, value, new CallArgs(pos, nm));
        }
        catch (TemplateException e) when (e.Annotate(Line, Col))
        {
            throw;
        }
    }
}
