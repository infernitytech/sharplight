using System.Text.Encodings.Web;
using System.Text.Json;

namespace Sharplight.Llm.Structured;

/// <summary>
/// Compiles a JSON schema into grammar rules for the JSON texts it accepts.
/// </summary>
/// <remarks>
/// Supported: <c>type</c> (single or list), <c>properties</c> / <c>required</c> (properties appear in schema order,
/// optional ones may be left out), <c>additionalProperties</c> (only for objects without <c>properties</c>; objects with
/// <c>properties</c> are closed, as OpenAI strict mode requires), <c>items</c>, <c>prefixItems</c>, <c>minItems</c> /
/// <c>maxItems</c>, <c>enum</c>, <c>const</c>, <c>anyOf</c> / <c>oneOf</c>, <c>allOf</c> (object members merged),
/// local <c>$ref</c> (<c>#</c>, <c>#/$defs/..</c>, <c>#/definitions/..</c>, recursion allowed), <c>nullable</c>,
/// <c>minLength</c> / <c>maxLength</c>, <c>pattern</c> (matched as a whole, without JSON escapes) and the formats
/// <c>date</c>, <c>time</c>, <c>date-time</c>, <c>uuid</c>. Numeric bounds other than a non-negative <c>minimum</c> are not
/// enforced; unknown keywords are ignored. Whitespace between tokens is limited to <see cref="MaxWhitespace"/> bytes so a
/// model cannot loop on blanks.
/// </remarks>
internal sealed class JsonSchemaCompiler
{
    /// <summary>Longest run of whitespace allowed between JSON tokens.</summary>
    public const int MaxWhitespace = 40;
    /// <summary>Bounded repetitions above this many items are treated as unbounded (each bound costs a grammar rule).</summary>
    private const int MaxExplicitBound = 512;

    private const string Ws = "json-ws";
    private const string Value = "json-value";
    private const string Object = "json-object";
    private const string Array = "json-array";
    private const string String = "json-string";
    private const string Char = "json-char";
    private const string Number = "json-number";
    private const string Integer = "json-integer";

    private static readonly JsonWriterOptions CompactWriter = new() { Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping };

    private readonly GrammarBuilder _builder;
    private readonly JsonElement _root;
    /// <summary>Prefix of this schema's rules, unique within the builder (several schemas may share one grammar).</summary>
    private readonly string _prefix;
    private int _rules;

    public JsonSchemaCompiler(GrammarBuilder builder, JsonElement root)
    {
        _builder = builder;
        _root = root;
        _prefix = builder.NewName("schema");
        DefineCommonRules(builder);
    }

    /// <summary>A whole document: the value with optional surrounding whitespace.</summary>
    public static GrammarExpr Document(GrammarExpr value) => GrammarExpr.Seq(GrammarExpr.Ref(Ws), value, GrammarExpr.Ref(Ws));

    /// <summary>Optional whitespace between JSON tokens (bounded, see <see cref="MaxWhitespace"/>); requires <see cref="DefineCommonRules"/>.</summary>
    public static GrammarExpr Whitespace() => GrammarExpr.Ref(Ws);

    /// <summary>Any JSON value.</summary>
    public static GrammarExpr AnyValue() => GrammarExpr.Ref(Value);

    /// <summary>Any JSON object.</summary>
    public static GrammarExpr AnyObject() => GrammarExpr.Ref(Object);

    /// <summary>Defines the shared JSON rules (value, object, array, string, number, whitespace) once per builder.</summary>
    public static void DefineCommonRules(GrammarBuilder builder)
    {
        if (builder.Contains(Value))
        {
            return;
        }

        var ws = GrammarExpr.Ref(Ws);
        builder.Add(Ws, GrammarExpr.Repeat(GrammarExpr.Bytes(ByteSet.Of((byte)' ').Union(ByteSet.Of((byte)'\t')).Union(ByteSet.Of((byte)'\n')).Union(ByteSet.Of((byte)'\r'))), 0, MaxWhitespace));
        builder.Add(Value, GrammarExpr.Choice(
            GrammarExpr.Ref(Object), GrammarExpr.Ref(Array), GrammarExpr.Ref(String), GrammarExpr.Ref(Number),
            GrammarExpr.Literal("true"), GrammarExpr.Literal("false"), GrammarExpr.Literal("null")));
        builder.Add(Object, ObjectOf(GrammarExpr.Ref(Value)));
        builder.Add(Array, GrammarExpr.Seq(
            GrammarExpr.Literal("["), ws,
            GrammarExpr.Optional(GrammarExpr.Seq(GrammarExpr.Ref(Value), GrammarExpr.Star(GrammarExpr.Seq(Separator(), GrammarExpr.Ref(Value))))),
            ws, GrammarExpr.Literal("]")));
        var plain = new CharClass().Add(0x20, 0x10FFFF);
        var unescaped = plain.Negate().Add('"').Add('\\').Negate();
        var hex = GrammarExpr.Bytes(ByteSet.Range('0', '9').Union(ByteSet.Range('a', 'f')).Union(ByteSet.Range('A', 'F')));
        builder.Add(Char, GrammarExpr.Choice(
            unescaped.ToExpr(),
            GrammarExpr.Seq(GrammarExpr.Literal("\\"), GrammarExpr.Choice(
                GrammarExpr.Bytes(Set("\"\\/bfnrt")),
                GrammarExpr.Seq(GrammarExpr.Literal("u"), GrammarExpr.Repeat(hex, 4, 4))))));
        builder.Add(String, Quoted(GrammarExpr.Star(GrammarExpr.Ref(Char))));
        var digit = GrammarExpr.Bytes(ByteSet.Range('0', '9'));
        var natural = GrammarExpr.Choice(GrammarExpr.Literal("0"), GrammarExpr.Seq(GrammarExpr.Bytes(ByteSet.Range('1', '9')), GrammarExpr.Repeat(digit, 0, 15)));
        builder.Add(Integer, GrammarExpr.Seq(GrammarExpr.Optional(GrammarExpr.Literal("-")), natural));
        builder.Add("json-natural", natural);
        builder.Add(Number, GrammarExpr.Seq(
            GrammarExpr.Ref(Integer),
            GrammarExpr.Optional(GrammarExpr.Seq(GrammarExpr.Literal("."), GrammarExpr.Repeat(digit, 1, 16))),
            GrammarExpr.Optional(GrammarExpr.Seq(GrammarExpr.Bytes(Set("eE")), GrammarExpr.Optional(GrammarExpr.Bytes(Set("+-"))), GrammarExpr.Repeat(digit, 1, 3)))));
    }

    private static ByteSet Set(string chars)
    {
        ByteSet set = default;
        foreach (char c in chars)
        {
            set = set.Union(ByteSet.Of((byte)c));
        }

        return set;
    }

    private static GrammarExpr Quoted(GrammarExpr body) => GrammarExpr.Seq(GrammarExpr.Literal("\""), body, GrammarExpr.Literal("\""));

    private static GrammarExpr Separator() => GrammarExpr.Seq(GrammarExpr.Ref(Ws), GrammarExpr.Literal(","), GrammarExpr.Ref(Ws));

    private static GrammarExpr ObjectOf(GrammarExpr value)
    {
        var ws = GrammarExpr.Ref(Ws);
        var member = GrammarExpr.Seq(GrammarExpr.Ref(String), ws, GrammarExpr.Literal(":"), ws, value);
        return GrammarExpr.Seq(
            GrammarExpr.Literal("{"), ws,
            GrammarExpr.Optional(GrammarExpr.Seq(member, GrammarExpr.Star(GrammarExpr.Seq(Separator(), member)))),
            ws, GrammarExpr.Literal("}"));
    }

    private static string QuotedName(string name) => "\"" + JsonEncodedText.Encode(name, JavaScriptEncoder.UnsafeRelaxedJsonEscaping) + "\"";

    /// <summary>The compact JSON text of <paramref name="value"/> (non-ASCII kept as is).</summary>
    internal static string Compact(JsonElement value)
    {
        using var stream = new MemoryStream();
        using (var writer = new Utf8JsonWriter(stream, CompactWriter))
        {
            value.WriteTo(writer);
        }

        return System.Text.Encoding.UTF8.GetString(stream.ToArray());
    }

    /// <summary>The expression for values valid under <paramref name="schema"/>.</summary>
    /// <exception cref="NotSupportedException">An unsupported construct (non-local <c>$ref</c>, invalid <c>pattern</c>).</exception>
    public GrammarExpr Compile(JsonElement schema)
    {
        if (schema.ValueKind == JsonValueKind.True || schema.ValueKind == JsonValueKind.Undefined)
        {
            return AnyValue();
        }

        if (schema.ValueKind != JsonValueKind.Object)
        {
            throw new NotSupportedException("A JSON schema must be an object or a boolean.");
        }

        var expr = CompileObjectSchema(schema);
        if (schema.TryGetProperty("nullable", out var nullable) && nullable.ValueKind == JsonValueKind.True)
        {
            expr = GrammarExpr.Choice(expr, GrammarExpr.Literal("null"));
        }

        return expr;
    }

    private GrammarExpr CompileObjectSchema(JsonElement schema)
    {
        if (schema.TryGetProperty("$ref", out var reference))
        {
            return Reference(reference.GetString() ?? "");
        }

        if (schema.TryGetProperty("const", out var constant))
        {
            return GrammarExpr.Literal(Compact(constant));
        }

        if (schema.TryGetProperty("enum", out var values) && values.ValueKind == JsonValueKind.Array)
        {
            return GrammarExpr.Choice(values.EnumerateArray().Select(v => GrammarExpr.Literal(Compact(v))));
        }

        foreach (string combinator in (ReadOnlySpan<string>)["anyOf", "oneOf"])
        {
            if (schema.TryGetProperty(combinator, out var options) && options.ValueKind == JsonValueKind.Array && options.GetArrayLength() > 0)
            {
                return GrammarExpr.Choice(options.EnumerateArray().Select(Compile));
            }
        }

        if (schema.TryGetProperty("allOf", out var all) && all.ValueKind == JsonValueKind.Array && all.GetArrayLength() > 0)
        {
            return CompileAllOf(schema, all);
        }

        if (schema.TryGetProperty("type", out var type))
        {
            if (type.ValueKind == JsonValueKind.Array)
            {
                return GrammarExpr.Choice(type.EnumerateArray().Select(t => CompileType(t.GetString() ?? "", schema)));
            }

            return CompileType(type.GetString() ?? "", schema);
        }

        if (schema.TryGetProperty("properties", out _) || schema.TryGetProperty("additionalProperties", out _))
        {
            return CompileType("object", schema);
        }

        if (schema.TryGetProperty("items", out _) || schema.TryGetProperty("prefixItems", out _))
        {
            return CompileType("array", schema);
        }

        return AnyValue();
    }

    private GrammarExpr CompileType(string type, JsonElement schema) => type switch
    {
        "object" => CompileObject(schema),
        "array" => CompileArray(schema),
        "string" => CompileString(schema),
        "integer" => NonNegative(schema) ? GrammarExpr.Ref("json-natural") : GrammarExpr.Ref(Integer),
        "number" => GrammarExpr.Ref(Number),
        "boolean" => GrammarExpr.Choice(GrammarExpr.Literal("true"), GrammarExpr.Literal("false")),
        "null" => GrammarExpr.Literal("null"),
        _ => throw new NotSupportedException($"Unknown JSON schema type '{type}'."),
    };

    private static bool NonNegative(JsonElement schema)
        => (schema.TryGetProperty("minimum", out var min) && min.ValueKind == JsonValueKind.Number && min.GetDouble() >= 0)
        || (schema.TryGetProperty("exclusiveMinimum", out var ex) && ex.ValueKind == JsonValueKind.Number && ex.GetDouble() >= -1);

    private GrammarExpr Reference(string reference)
    {
        if (!reference.StartsWith('#'))
        {
            throw new NotSupportedException($"Only local schema references are supported, not '{reference}'.");
        }

        string name = $"{_prefix}-ref:{reference}";
        if (_builder.Contains(name))
        {
            return GrammarExpr.Ref(name);
        }

        var target = _root;
        foreach (string part in reference[1..].Split('/', StringSplitOptions.RemoveEmptyEntries))
        {
            string key = Uri.UnescapeDataString(part).Replace("~1", "/").Replace("~0", "~");
            if (target.ValueKind != JsonValueKind.Object || !target.TryGetProperty(key, out target))
            {
                throw new NotSupportedException($"The schema reference '{reference}' does not resolve.");
            }
        }

        _builder.Add(name, GrammarExpr.Empty);   // placeholder: lets recursive schemas refer to themselves
        _builder.Add(name, Compile(target));
        return GrammarExpr.Ref(name);
    }

    private GrammarExpr CompileAllOf(JsonElement schema, JsonElement all)
    {
        // Merges object members of all parts (the common use: composing object schemas); anything else uses the first part.
        var properties = new List<JsonProperty>();
        var required = new HashSet<string>(StringComparer.Ordinal);
        bool objects = true;
        foreach (var (part, own) in all.EnumerateArray().Select(p => (p, false)).Prepend((schema, true)))
        {
            var resolved = Resolve(part);
            if (resolved.TryGetProperty("properties", out var props) && props.ValueKind == JsonValueKind.Object)
            {
                properties.AddRange(props.EnumerateObject().Where(p => properties.All(q => q.Name != p.Name)));
            }
            else if (!own && resolved.TryGetProperty("type", out var t) && t.ValueKind == JsonValueKind.String && t.GetString() != "object")
            {
                objects = false;
            }

            if (resolved.TryGetProperty("required", out var req) && req.ValueKind == JsonValueKind.Array)
            {
                required.UnionWith(req.EnumerateArray().Select(r => r.GetString() ?? ""));
            }
        }

        return objects ? ObjectWithProperties(properties, required) : Compile(all[0]);
    }

    private JsonElement Resolve(JsonElement schema)
    {
        while (schema.ValueKind == JsonValueKind.Object && schema.TryGetProperty("$ref", out var reference) && reference.GetString() is { } r && r.StartsWith('#'))
        {
            var target = _root;
            foreach (string part in r[1..].Split('/', StringSplitOptions.RemoveEmptyEntries))
            {
                if (!target.TryGetProperty(part, out target))
                {
                    throw new NotSupportedException($"The schema reference '{r}' does not resolve.");
                }
            }

            schema = target;
        }

        return schema;
    }

    private GrammarExpr CompileObject(JsonElement schema)
    {
        if (schema.TryGetProperty("properties", out var props) && props.ValueKind == JsonValueKind.Object && props.EnumerateObject().Any())
        {
            var required = new HashSet<string>(StringComparer.Ordinal);
            if (schema.TryGetProperty("required", out var req) && req.ValueKind == JsonValueKind.Array)
            {
                required.UnionWith(req.EnumerateArray().Select(r => r.GetString() ?? ""));
            }

            return ObjectWithProperties([.. props.EnumerateObject()], required);
        }

        if (schema.TryGetProperty("additionalProperties", out var additional))
        {
            if (additional.ValueKind == JsonValueKind.False)
            {
                return GrammarExpr.Seq(GrammarExpr.Literal("{"), GrammarExpr.Ref(Ws), GrammarExpr.Literal("}"));
            }

            if (additional.ValueKind == JsonValueKind.Object)
            {
                return ObjectOf(Compile(additional));
            }
        }

        return AnyObject();
    }

    /// <summary>
    /// A closed object with the given members in order. Optional members make the comma placement depend on whether a member
    /// was already written, so the tail after member i exists in two variants (first / not first), each a rule.
    /// </summary>
    private GrammarExpr ObjectWithProperties(IReadOnlyList<JsonProperty> properties, HashSet<string> required)
    {
        var ws = GrammarExpr.Ref(Ws);
        var members = properties.Select(p => GrammarExpr.Seq(
            GrammarExpr.Literal(QuotedName(p.Name)), ws, GrammarExpr.Literal(":"), ws, Compile(p.Value))).ToArray();
        int n = members.Length;
        string prefix = $"{_prefix}-obj{_rules++}";
        GrammarExpr Tail(int i, bool first) => i == n ? GrammarExpr.Empty : GrammarExpr.Ref($"{prefix}-{i}{(first ? "f" : "")}");
        for (int i = n - 1; i >= 0; i--)
        {
            foreach (bool first in (ReadOnlySpan<bool>)[true, false])
            {
                var take = GrammarExpr.Seq(first ? GrammarExpr.Empty : Separator(), members[i], Tail(i + 1, false));
                var body = required.Contains(properties[i].Name) ? take : GrammarExpr.Choice(take, Tail(i + 1, first));
                _builder.Add($"{prefix}-{i}{(first ? "f" : "")}", body);
            }
        }

        return GrammarExpr.Seq(GrammarExpr.Literal("{"), ws, Tail(0, true), ws, GrammarExpr.Literal("}"));
    }

    private GrammarExpr CompileArray(JsonElement schema)
    {
        var ws = GrammarExpr.Ref(Ws);
        int min = schema.TryGetProperty("minItems", out var mi) && mi.TryGetInt32(out int a) ? Math.Max(0, a) : 0;
        int max = schema.TryGetProperty("maxItems", out var ma) && ma.TryGetInt32(out int b) ? b : -1;
        if (max > MaxExplicitBound)
        {
            max = -1;
        }

        min = Math.Min(min, MaxExplicitBound);
        var prefix = new List<GrammarExpr>();
        if (schema.TryGetProperty("prefixItems", out var prefixItems) && prefixItems.ValueKind == JsonValueKind.Array)
        {
            prefix.AddRange(prefixItems.EnumerateArray().Select(Compile));
        }

        GrammarExpr? item = AnyValue();
        if (schema.TryGetProperty("items", out var items))
        {
            item = items.ValueKind == JsonValueKind.False ? null : Compile(items);
        }

        GrammarExpr body;
        if (prefix.Count == 0)
        {
            body = item is null || max == 0 ? GrammarExpr.Empty : Items(item, min, max);
        }
        else
        {
            // A tuple: the prefix items in order (all required), then further items as allowed.
            var parts = new List<GrammarExpr> { prefix[0] };
            foreach (var p in prefix.Skip(1))
            {
                parts.Add(Separator());
                parts.Add(p);
            }

            int restMax = max < 0 ? -1 : Math.Max(0, max - prefix.Count);
            if (item is not null && restMax != 0)
            {
                int restMin = Math.Max(0, min - prefix.Count);
                parts.Add(GrammarExpr.Repeat(GrammarExpr.Seq(Separator(), item), restMin, restMax));
            }

            body = GrammarExpr.Seq(parts);
        }

        return GrammarExpr.Seq(GrammarExpr.Literal("["), ws, body, ws, GrammarExpr.Literal("]"));
    }

    private GrammarExpr Items(GrammarExpr item, int min, int max)
    {
        string name = $"{_prefix}-item{_rules++}";
        _builder.Add(name, item);
        var unit = GrammarExpr.Ref(name);
        var more = GrammarExpr.Seq(Separator(), unit);
        int restMax = max < 0 ? -1 : max - 1;
        var nonEmpty = GrammarExpr.Seq(unit, GrammarExpr.Repeat(more, Math.Max(0, min - 1), restMax));
        return min == 0 ? GrammarExpr.Optional(nonEmpty) : nonEmpty;
    }

    private GrammarExpr CompileString(JsonElement schema)
    {
        if (schema.TryGetProperty("pattern", out var pattern) && pattern.GetString() is { } regex)
        {
            try
            {
                return Quoted(RegexCompiler.Compile(regex));
            }
            catch (FormatException e)
            {
                throw new NotSupportedException($"Unsupported string pattern: {e.Message}", e);
            }
        }

        if (schema.TryGetProperty("format", out var format) && FormatPattern(format.GetString()) is { } formatRegex)
        {
            return Quoted(RegexCompiler.Compile(formatRegex));
        }

        int min = schema.TryGetProperty("minLength", out var mi) && mi.TryGetInt32(out int a) ? Math.Clamp(a, 0, MaxExplicitBound) : 0;
        int max = schema.TryGetProperty("maxLength", out var ma) && ma.TryGetInt32(out int b) && b <= MaxExplicitBound ? b : -1;
        if (min == 0 && max < 0)
        {
            return GrammarExpr.Ref(String);
        }

        return Quoted(GrammarExpr.Repeat(GrammarExpr.Ref(Char), min, max));
    }

    private static string? FormatPattern(string? format) => format switch
    {
        "date" => @"[0-9]{4}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])",
        "time" => @"([01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9](\.[0-9]{1,9})?(Z|[+-]([01][0-9]|2[0-3]):[0-5][0-9])?",
        "date-time" => @"[0-9]{4}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])T([01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9](\.[0-9]{1,9})?(Z|[+-]([01][0-9]|2[0-3]):[0-5][0-9])",
        "uuid" => "[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
        _ => null,
    };
}
