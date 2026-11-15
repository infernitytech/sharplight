namespace Sharplight.Templating;

/// <summary>Lexical scope used while rendering.</summary>
internal sealed class Frame
{
    public readonly Dictionary<string, object?> Vars;
    public readonly Frame? Parent;
    public readonly JinjaEnvironment Env;
    public int LoopDepth;

    public Frame(JinjaEnvironment env, Frame? parent, Dictionary<string, object?>? vars = null)
    {
        Env = env;
        Parent = parent;
        Vars = vars ?? new Dictionary<string, object?>();
        LoopDepth = parent?.LoopDepth ?? 0;
    }

    public bool Strict => Env.StrictUndefined;

    public object? Lookup(string name)
    {
        for (var f = this; f != null; f = f.Parent)
        {
            if (f.Vars.TryGetValue(name, out var v))
            {
                return v;
            }
        }

        if (Env.Globals.TryGetValue(name, out var g))
        {
            return g;
        }

        return new Undefined(name, Strict);
    }

    public void Set(string name, object? value) => Vars[name] = value;
}
