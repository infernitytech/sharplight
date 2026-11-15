using System.Text.Json;

namespace Sharplight.Llm.Structured;

/// <summary>
/// A requested output structure (OpenAI <c>response_format</c>, vLLM <c>structured_outputs</c>): a JSON object, a JSON
/// schema, a regular expression, a GBNF grammar or a fixed set of choices. The chat layer compiles it into a grammar,
/// combined with the model's reasoning block and tool-call syntax where needed, and enforces it with token masks.
/// Immutable.
/// </summary>
public sealed class OutputFormat
{
    private enum FormatKind
    {
        JsonObject,
        JsonSchema,
        Regex,
        Gbnf,
        Choice,
    }

    private readonly FormatKind _kind;
    private readonly string _text;
    private readonly IReadOnlyList<string> _choices;

    private OutputFormat(FormatKind kind, string text, IReadOnlyList<string>? choices = null)
    {
        _kind = kind;
        _text = text;
        _choices = choices ?? [];
    }

    /// <summary>Any JSON object.</summary>
    public static OutputFormat JsonObject { get; } = new(FormatKind.JsonObject, "");

    /// <summary>JSON valid under <paramref name="schemaJson"/> (see <see cref="Grammar.FromJsonSchema(string)"/> for the supported subset).</summary>
    public static OutputFormat JsonSchema(string schemaJson) => new(FormatKind.JsonSchema, schemaJson);

    /// <summary>Text matching the regular expression as a whole (see <see cref="Grammar.FromRegex"/>).</summary>
    public static OutputFormat Regex(string pattern) => new(FormatKind.Regex, pattern);

    /// <summary>Text matching a GBNF grammar from its <c>root</c> rule (see <see cref="Grammar.FromGbnf"/>).</summary>
    public static OutputFormat Gbnf(string grammar) => new(FormatKind.Gbnf, grammar);

    /// <summary>Exactly one of <paramref name="choices"/>.</summary>
    public static OutputFormat Choice(IReadOnlyList<string> choices) => new(FormatKind.Choice, "", choices);

    /// <summary>True for the JSON formats (the chat layer adds a JSON instruction to the prompt for them).</summary>
    public bool IsJson => _kind is FormatKind.JsonObject or FormatKind.JsonSchema;

    /// <summary>Identifies the format for caching compiled grammars.</summary>
    internal string CacheKey => $"{_kind}\u0001{_text}\u0001{string.Join('\u0001', _choices)}";

    /// <summary>The schema text of a JSON-schema format, else null.</summary>
    internal string? Schema => _kind == FormatKind.JsonSchema ? _text : null;

    /// <summary>Compiles the format alone into a grammar, validating it.</summary>
    /// <exception cref="FormatException">An invalid regex or grammar.</exception>
    /// <exception cref="ArgumentException">An invalid grammar (undefined or left-recursive rules) or an empty choice list.</exception>
    /// <exception cref="JsonException">The schema is not JSON.</exception>
    /// <exception cref="NotSupportedException">The schema uses an unsupported construct.</exception>
    public Grammar ToGrammar()
    {
        var builder = new GrammarBuilder();
        builder.Add("root", AddTo(builder));
        return builder.Build("root");
    }

    /// <summary>Adds the rules of this format to <paramref name="builder"/> and returns the expression of the formatted text.</summary>
    internal GrammarExpr AddTo(GrammarBuilder builder)
    {
        switch (_kind)
        {
            case FormatKind.JsonObject:
                JsonSchemaCompiler.DefineCommonRules(builder);
                return JsonSchemaCompiler.Document(JsonSchemaCompiler.AnyObject());
            case FormatKind.JsonSchema:
            {
                using var doc = JsonDocument.Parse(_text);
                var compiler = new JsonSchemaCompiler(builder, doc.RootElement);
                return JsonSchemaCompiler.Document(compiler.Compile(doc.RootElement));
            }

            case FormatKind.Regex:
                return RegexCompiler.Compile(_text);
            case FormatKind.Gbnf:
            {
                string prefix = builder.NewName("gbnf") + ":";
                GbnfParser.ParseInto(builder, _text, prefix);
                if (!builder.Contains(prefix + "root"))
                {
                    throw new ArgumentException("The grammar has no 'root' rule.");
                }

                return GrammarExpr.Ref(prefix + "root");
            }

            default:
                if (_choices.Count == 0)
                {
                    throw new ArgumentException("A choice format needs at least one choice.");
                }

                return GrammarExpr.Choice(_choices.Select(GrammarExpr.Literal));
        }
    }
}
