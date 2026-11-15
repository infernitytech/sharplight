using System.Text;

namespace Sharplight.Templating;

/// <summary>A compiled template. Thread-safe for concurrent <see cref="Render"/> calls as long as the context values are not shared mutable lists.</summary>
public sealed class JinjaTemplate
{
    private readonly JinjaEnvironment _env;
    private readonly List<Stmt> _body;

    internal JinjaTemplate(JinjaEnvironment env, List<Stmt> body)
    {
        _env = env;
        _body = body;
    }

    /// <summary>Renders the template with the given top-level variables (values in the template value model, see <see cref="JinjaValue.From"/>).</summary>
    /// <exception cref="TemplateRuntimeException">Rendering failed.</exception>
    public string Render(IReadOnlyDictionary<string, object?>? context = null)
    {
        var vars = new Dictionary<string, object?>();
        if (context != null)
        {
            foreach (var kv in context)
            {
                vars[kv.Key] = JinjaValue.From(kv.Value);
            }
        }

        var frame = new Frame(_env, null, vars);
        var sb = new StringBuilder();
        Stmt.ExecAll(_body, frame, sb);
        return sb.ToString();
    }
}
