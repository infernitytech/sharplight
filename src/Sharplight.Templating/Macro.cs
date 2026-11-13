using System.Text;

namespace Sharplight.Templating;

internal sealed class Macro(MacroDef def, Frame closure) : JinjaCallable
{
    public override string Name => def.Name;

    public override object? Invoke(CallArgs args)
    {
        var frame = new Frame(closure.Env, closure);
        var ps = def.Parameters;
        for (int i = 0; i < ps.Count; i++)
        {
            var (pn, pd) = ps[i];
            if (i < args.Positional.Count)
            {
                frame.Vars[pn] = args.Positional[i];
            }
            else if (args.Named != null && args.Named.TryGetValue(pn, out var nv))
            {
                frame.Vars[pn] = nv;
            }
            else if (pd != null)
            {
                frame.Vars[pn] = pd.Eval(frame);
            }
            else
            {
                frame.Vars[pn] = new Undefined(pn, closure.Strict);
            }
        }
        var varargs = new JinjaTuple();
        for (int i = ps.Count; i < args.Positional.Count; i++)
        {
            varargs.Add(args.Positional[i]);
        }

        var kwargs = new JinjaDict();
        if (args.Named != null)
        {
            foreach (var (k, v) in args.Named)
            {
                if (k == "caller")
                {
                    frame.Vars["caller"] = v;
                }
                else if (!ps.Exists(p => p.Name == k))
                {
                    kwargs[k] = v;
                }
            }
        }
        frame.Vars["varargs"] = varargs;
        frame.Vars["kwargs"] = kwargs;
        var sb = new StringBuilder();
        Stmt.ExecAll(def.Body, frame, sb);
        return sb.ToString();
    }
}
