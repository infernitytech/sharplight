namespace Sharplight.Templating;

internal abstract class Expr : Node
{
    public object? Eval(Frame f)
    {
        try
        {
            return Core(f);
        }
        catch (TemplateException e) when (e.Annotate(Line, Col))
        {
            throw;
        }
    }

    protected abstract object? Core(Frame f);
    public virtual string Describe() => "expression";
}
