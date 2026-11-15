namespace Sharplight.Templating;

internal sealed class MacroDef(string name, List<(string Name, Expr? Default)> parameters, List<Stmt> body)
{
    public string Name => name;
    public List<(string Name, Expr? Default)> Parameters => parameters;
    public List<Stmt> Body => body;
}
