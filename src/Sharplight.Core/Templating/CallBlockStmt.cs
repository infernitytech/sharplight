using System.Text;

namespace Sharplight.Templating;

internal sealed class CallBlockStmt(MacroDef caller, CallExpr call) : Stmt
{
    public override Flow Exec(Frame f, StringBuilder o)
    {
        o.Append(Ops.Str(call.EvalWithCaller(f, new Macro(caller, f))));
        return Flow.Normal;
    }
}
