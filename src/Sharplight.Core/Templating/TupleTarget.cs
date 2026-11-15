namespace Sharplight.Templating;

internal sealed class TupleTarget(List<Target> items) : Target
{
    public override void Assign(Frame f, object? value)
    {
        if (value is Undefined u && u.Strict)
        {
            throw u.Fail();
        }

        if (!Ops.IsIterable(value))
        {
            throw new TemplateRuntimeException($"cannot unpack non-iterable {Ops.TypeName(value)} object");
        }

        var l = Ops.Iterate(value);
        if (l.Count != items.Count)
        {
            throw new TemplateRuntimeException(l.Count < items.Count
                ? $"not enough values to unpack (expected {items.Count}, got {l.Count})"
                : $"too many values to unpack (expected {items.Count})");
        }

        for (int i = 0; i < items.Count; i++)
        {
            items[i].Assign(f, l[i]);
        }
    }
}

// ============================================================================ statements
