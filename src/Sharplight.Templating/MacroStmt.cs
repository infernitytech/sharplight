using System.Text;

namespace Sharplight.Templating;

internal sealed class MacroStmt(MacroDef def) : Stmt
{
    public override Flow Exec(Frame f, StringBuilder o)
    {
        f.Set(def.Name, new Macro(def, f));
        return Flow.Normal;
    }
}
