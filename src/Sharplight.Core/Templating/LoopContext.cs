namespace Sharplight.Templating;

internal sealed class LoopContext
{
    public List<object?> Items = new();
    public int Index0 = -1;
    public int Depth0 => 0;
    private object? _changedLast = Unset;
    private static readonly object Unset = new();

    public int Length => Items.Count;

    public object? Get(string name, bool strict)
    {
        switch (name)
        {
            case "index": return (long)(Index0 + 1);
            case "index0": return (long)Index0;
            case "revindex": return (long)(Length - Index0);
            case "revindex0": return (long)(Length - Index0 - 1);
            case "first": return Index0 == 0;
            case "last": return Index0 == Length - 1;
            case "length": return (long)Length;
            case "depth": return (long)(Depth0 + 1);
            case "depth0": return (long)Depth0;
            case "previtem": return Index0 > 0 ? Items[Index0 - 1] : new Undefined("loop.previtem", strict);
            case "nextitem": return Index0 < Length - 1 ? Items[Index0 + 1] : new Undefined("loop.nextitem", strict);
            case "cycle":
            case "changed": return new BoundMethod(this, name);
            default: return new Undefined("loop." + name, strict);
        }
    }

    public object? Call(string name, CallArgs args)
    {
        if (name == "cycle")
        {
            if (args.Positional.Count == 0)
            {
                throw new TemplateRuntimeException("loop.cycle() requires at least one argument");
            }

            return args.Positional[Index0 % args.Positional.Count];
        }
        // changed
        var val = (object?)new List<object?>(args.Positional);
        bool changed = ReferenceEquals(_changedLast, Unset) || !Ops.Eq(_changedLast, val);
        _changedLast = val;
        return changed;
    }
}
