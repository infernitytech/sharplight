namespace Sharplight.Templating;

internal sealed class DictExpr(List<(Expr Key, Expr Value)> items) : Expr
{
    protected override object? Core(Frame f)
    {
        var d = new JinjaDict();
        foreach (var (k, v) in items)
        {
            var key = k.Eval(f);
            if (key is not (string or long or bool or double))
            {
                throw new TemplateRuntimeException($"unhashable or unsupported dict key of type '{Ops.TypeName(key)}'");
            }

            d[key is double dk && dk == Math.Floor(dk) ? (long)dk : key!] = v.Eval(f);
        }
        return d;
    }
}
