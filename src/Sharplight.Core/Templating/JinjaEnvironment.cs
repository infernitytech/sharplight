namespace Sharplight.Templating;

/// <summary>Configuration shared by templates: whitespace options, undefined behavior, globals, filters and tests.</summary>
public sealed class JinjaEnvironment
{
    /// <summary>Creates an environment with the built-in filters and tests and default options.</summary>
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

    /// <summary>Filters by name (<c>{{ x | name }}</c>); add entries to register custom filters. Not thread-safe to mutate while rendering.</summary>
    public Dictionary<string, FilterFunction> Filters { get; }
    /// <summary>Tests by name (<c>{% if x is name %}</c>); add entries to register custom tests.</summary>
    public Dictionary<string, TestFunction> Tests { get; }

    /// <summary>Parses a template. Throws <see cref="TemplateSyntaxException"/> on errors.</summary>
    public JinjaTemplate Compile(string source)
    {
        var tokens = Lexer.Tokenize(source, TrimBlocks, LstripBlocks, KeepTrailingNewline);
        var body = Parser.Parse(tokens);
        return new JinjaTemplate(this, body);
    }
}
