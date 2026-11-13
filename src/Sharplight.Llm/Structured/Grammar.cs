using System.Text.Json;

namespace Sharplight.Llm.Structured;

/// <summary>
/// A compiled, byte-level context-free grammar for constrained decoding. Build one from GBNF text (the llama.cpp format),
/// a regular expression or a JSON schema, then bind it to a tokenizer with <see cref="GrammarConstraint"/>.
/// Immutable and thread-safe; compile once and reuse.
/// </summary>
/// <remarks>
/// The grammar matches UTF-8 bytes. Character classes are compiled into exact UTF-8 byte sequences; left-recursive rules
/// are rejected.
/// </remarks>
/// <example>
/// <code>
/// var grammar = Grammar.FromJsonSchema("""{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}""");
/// var request = new GenerationRequest { PromptTokens = prompt, Constraint = new GrammarConstraint(grammar, tokenizer) };
/// </code>
/// </example>
public sealed class Grammar
{
    internal enum ElementKind : byte
    {
        Bytes,
        Rule,
        End,
    }

    /// <summary>One position of an alternative: a byte set to match, a rule to enter, or the end of the alternative.</summary>
    internal readonly record struct Element(ElementKind Kind, ByteSet Set, int Rule);

    internal Grammar(Element[] elements, int[][] alternatives, string[] ruleNames)
    {
        Elements = elements;
        Alternatives = alternatives;
        RuleNames = ruleNames;
    }

    /// <summary>All alternatives of all rules, each closed by an <see cref="ElementKind.End"/> element.</summary>
    internal Element[] Elements { get; }
    /// <summary>Start positions in <see cref="Elements"/> of each rule's alternatives; rule 0 is the root.</summary>
    internal int[][] Alternatives { get; }
    internal string[] RuleNames { get; }

    /// <summary>Number of rules, including the helper rules generated for repetitions and nested alternatives.</summary>
    public int RuleCount => Alternatives.Length;

    /// <summary>Parses a grammar in the GBNF format of llama.cpp (<c>root ::= ...</c>).</summary>
    /// <param name="gbnf">The grammar text.</param>
    /// <param name="root">The start rule.</param>
    /// <exception cref="FormatException">The text is not valid GBNF.</exception>
    /// <exception cref="ArgumentException">A rule is undefined or left-recursive.</exception>
    public static Grammar FromGbnf(string gbnf, string root = "root") => GbnfParser.Parse(gbnf).Build(root);

    /// <summary>A grammar matching exactly the strings of a regular expression (the whole output must match).</summary>
    /// <param name="pattern">
    /// The pattern: literals, escapes (<c>\d \w \s \xHH \uHHHH</c>...), classes, <c>.</c>, groups, alternation and the
    /// quantifiers <c>* + ? {n} {n,} {n,m}</c>. Anchors, word boundaries, lazy/possessive markers are accepted and ignored;
    /// back-references and look-arounds are not supported.
    /// </param>
    /// <exception cref="FormatException">The pattern is invalid or uses an unsupported feature.</exception>
    public static Grammar FromRegex(string pattern)
    {
        var builder = new GrammarBuilder();
        builder.Add("root", RegexCompiler.Compile(pattern));
        return builder.Build("root");
    }

    /// <summary>A grammar for JSON documents valid under a JSON schema (see <see cref="JsonSchemaCompiler"/> for the subset).</summary>
    /// <param name="schemaJson">The schema as JSON text.</param>
    /// <exception cref="JsonException">The text is not JSON.</exception>
    /// <exception cref="NotSupportedException">The schema uses an unsupported construct (for example an external <c>$ref</c>).</exception>
    public static Grammar FromJsonSchema(string schemaJson) => OutputFormat.JsonSchema(schemaJson).ToGrammar();

    /// <summary>A grammar for JSON documents valid under a JSON schema.</summary>
    /// <param name="schema">The schema.</param>
    /// <exception cref="NotSupportedException">The schema uses an unsupported construct.</exception>
    public static Grammar FromJsonSchema(JsonElement schema)
    {
        var builder = new GrammarBuilder();
        var compiler = new JsonSchemaCompiler(builder, schema);
        builder.Add("root", JsonSchemaCompiler.Document(compiler.Compile(schema)));
        return builder.Build("root");
    }

    /// <summary>Any JSON object (OpenAI <c>response_format: {"type": "json_object"}</c>).</summary>
    public static Grammar JsonObject { get; } = OutputFormat.JsonObject.ToGrammar();

    /// <exception cref="ArgumentException">A rule can reach itself without consuming input.</exception>
    internal void CheckNoLeftRecursion()
    {
        int rules = Alternatives.Length;
        var nullable = new bool[rules];
        bool changed = true;
        while (changed)
        {
            changed = false;
            for (int r = 0; r < rules; r++)
            {
                if (!nullable[r] && Alternatives[r].Any(start => PrefixNullable(start, nullable)))
                {
                    nullable[r] = true;
                    changed = true;
                }
            }
        }

        // Rule r "left-calls" every rule referenced before its first non-nullable element.
        var state = new byte[rules];   // 0 unvisited, 1 on the stack, 2 done
        for (int r = 0; r < rules; r++)
        {
            Visit(r, nullable, state);
        }
    }

    private void Visit(int rule, bool[] nullable, byte[] state)
    {
        if (state[rule] == 2)
        {
            return;
        }

        if (state[rule] == 1)
        {
            throw new ArgumentException($"The grammar is left-recursive at rule '{RuleNames[rule]}'.");
        }

        state[rule] = 1;
        foreach (int start in Alternatives[rule])
        {
            for (int pos = start; Elements[pos].Kind == ElementKind.Rule; pos++)
            {
                Visit(Elements[pos].Rule, nullable, state);
                if (!nullable[Elements[pos].Rule])
                {
                    break;
                }
            }
        }

        state[rule] = 2;
    }

    /// <summary>True when the alternative starting at <paramref name="start"/> can match the empty string.</summary>
    private bool PrefixNullable(int start, bool[] nullable)
    {
        for (int pos = start; ; pos++)
        {
            var e = Elements[pos];
            if (e.Kind == ElementKind.End)
            {
                return true;
            }

            if (e.Kind == ElementKind.Bytes || !nullable[e.Rule])
            {
                return false;
            }
        }
    }
}
