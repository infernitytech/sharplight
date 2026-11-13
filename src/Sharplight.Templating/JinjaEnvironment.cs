using System.Text;

namespace Sharplight.Templating;

/// <summary>Configuration shared by templates: whitespace options, undefined behavior, globals, filters and tests.</summary>
public sealed class JinjaEnvironment
{
    public JinjaEnvironment()
    {
        Filters = Builtins.CreateFilters();
        Tests = Builtins.CreateTests();
        Builtins.AddGlobals(this);
    }

    /// <summary>Remove the first newline after a block tag.</summary>
    public bool TrimBlocks { get; set; }

    /// <summary>Strip whitespace before a block tag that starts a line.</summary>
    public bool LstripBlocks { get; set; }

    /// <summary>Keep the single trailing newline of the template source (Jinja strips it by default).</summary>
    public bool KeepTrailingNewline { get; set; }

    /// <summary>If true, using an undefined value (printing, comparing, iterating, attribute access) throws.</summary>
    public bool StrictUndefined { get; set; }

    /// <summary>Time source for <c>strftime_now</c>.</summary>
    public Func<DateTimeOffset> Clock { get; set; } = () => DateTimeOffset.Now;

    /// <summary>Global variables; values must be in the template value model (see <see cref="JinjaValue.From"/>).</summary>
    public Dictionary<string, object?> Globals { get; } = new();

    public Dictionary<string, FilterFunction> Filters { get; }
    public Dictionary<string, TestFunction> Tests { get; }

    /// <summary>Parses a template. Throws <see cref="TemplateSyntaxException"/> on errors.</summary>
    public JinjaTemplate Compile(string source)
    {
        var tokens = Lexer.Tokenize(source, TrimBlocks, LstripBlocks, KeepTrailingNewline);
        var body = Parser.Parse(tokens);
        return new JinjaTemplate(this, body);
    }
}

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

    public string Render(IReadOnlyDictionary<string, object?>? context = null)
    {
        var vars = new Dictionary<string, object?>();
        if (context != null)
            foreach (var kv in context) vars[kv.Key] = JinjaValue.From(kv.Value);
        var frame = new Frame(_env, null, vars);
        var sb = new StringBuilder();
        Stmt.ExecAll(_body, frame, sb);
        return sb.ToString();
    }
}
