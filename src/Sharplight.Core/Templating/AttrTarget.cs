namespace Sharplight.Templating;

internal sealed class AttrTarget(string objName, string attr) : Target
{
    public override void Assign(Frame f, object? value)
    {
        var o = f.Lookup(objName);
        if (o is JinjaNamespace ns)
        {
            ns.Attributes[attr] = value;
        }
        else
        {
            throw new TemplateRuntimeException($"cannot assign attribute '{attr}' on '{objName}' (type {Ops.TypeName(o)}); only namespace objects support attribute assignment");
        }
    }
}
