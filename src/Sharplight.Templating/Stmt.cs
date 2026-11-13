using System.Text;

namespace Sharplight.Templating;

internal abstract class Stmt : Node
{
    public abstract Flow Exec(Frame f, StringBuilder o);

    public static Flow ExecAll(List<Stmt> body, Frame f, StringBuilder o)
    {
        foreach (var s in body)
        {
            var flow = s.Exec(f, o);
            if (flow != Flow.Normal)
            {
                return flow;
            }
        }
        return Flow.Normal;
    }
}

// ============================================================================ expressions
