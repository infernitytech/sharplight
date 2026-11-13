using System.Text.Json;
using System.Text.RegularExpressions;
using Microsoft.Extensions.AI;

namespace Sharplight.Extensions.AI;

/// <summary>Turns tool-call text emitted by the model into a <see cref="FunctionCallContent"/>.</summary>
public interface IToolCallParser
{
    /// <summary>Parses the text between the tool-call start and end markers; returns null if it is not this parser's format.</summary>
    public FunctionCallContent? TryParse(string body, ToolSchemas tools);
}

/// <summary>The tools offered in a request, used to coerce text parameter values to their declared JSON types.</summary>
public sealed class ToolSchemas
{
    public static readonly ToolSchemas Empty = new([]);
    private readonly Dictionary<string, JsonElement> _parameters = new();
    private readonly HashSet<string> _names = new();

    public ToolSchemas(IEnumerable<AITool>? tools)
    {
        if (tools == null)
        {
            return;
        }

        foreach (var t in tools)
        {
            if (t is not AIFunctionDeclaration f)
            {
                continue;
            }

            _names.Add(f.Name);
            if (f.JsonSchema.ValueKind == JsonValueKind.Object && f.JsonSchema.TryGetProperty("properties", out var props) && props.ValueKind == JsonValueKind.Object)
            {
                _parameters[f.Name] = props;
            }
        }
    }

    public bool Has(string name) => _names.Contains(name);

    /// <summary>The declared JSON schema "type" of a parameter, if known.</summary>
    public string? ParameterType(string function, string parameter)
    {
        if (_parameters.TryGetValue(function, out var props) && props.TryGetProperty(parameter, out var p)
            && p.ValueKind == JsonValueKind.Object && p.TryGetProperty("type", out var t) && t.ValueKind == JsonValueKind.String)
        {
            return t.GetString();
        }

        return null;
    }
}

/// <summary>Hermes / Qwen2.5 / Qwen3 style: <c>{"name": "f", "arguments": {...}}</c> inside the tool-call tags.</summary>
public sealed class JsonToolCallParser : IToolCallParser
{
    public FunctionCallContent? TryParse(string body, ToolSchemas tools)
    {
        body = body.Trim();
        if (!body.StartsWith('{'))
        {
            return null;
        }

        try
        {
            using var doc = JsonDocument.Parse(body);
            var root = doc.RootElement;
            if (root.ValueKind != JsonValueKind.Object)
            {
                return null;
            }

            if (root.TryGetProperty("function", out var fn) && fn.ValueKind == JsonValueKind.Object)
            {
                root = fn;
            }

            if (!root.TryGetProperty("name", out var n) || n.ValueKind != JsonValueKind.String)
            {
                return null;
            }

            Dictionary<string, object?>? args = null;
            if (root.TryGetProperty("arguments", out var a) || root.TryGetProperty("parameters", out a))
            {
                if (a.ValueKind == JsonValueKind.String)
                {
                    using var inner = JsonDocument.Parse(a.GetString()!);
                    args = ToArguments(inner.RootElement);
                }
                else
                {
                    args = ToArguments(a);
                }
            }
            return new FunctionCallContent(ToolCallIds.New(), n.GetString()!, args ?? []);
        }
        catch (JsonException) { return null; }
    }

    private static Dictionary<string, object?> ToArguments(JsonElement e)
    {
        var d = new Dictionary<string, object?>();
        if (e.ValueKind != JsonValueKind.Object)
        {
            return d;
        }

        foreach (var p in e.EnumerateObject())
        {
            d[p.Name] = p.Value.Clone();
        }

        return d;
    }
}

/// <summary>
/// Qwen3.5 / Qwen3-Coder style:
/// <c>&lt;function=name&gt;&lt;parameter=key&gt;value&lt;/parameter&gt;&lt;/function&gt;</c> inside the tool-call tags.
/// Parameter values are coerced to the declared schema type (string values stay strings).
/// </summary>
public sealed partial class XmlToolCallParser : IToolCallParser
{
    [GeneratedRegex(@"<function=([^>\s]+)>", RegexOptions.CultureInvariant)]
    private static partial Regex FunctionRegex();
    [GeneratedRegex(@"<parameter=([^>\s]+)>(.*?)</parameter>", RegexOptions.CultureInvariant | RegexOptions.Singleline)]
    private static partial Regex ParameterRegex();

    public FunctionCallContent? TryParse(string body, ToolSchemas tools)
    {
        var fm = FunctionRegex().Match(body);
        if (!fm.Success)
        {
            return null;
        }

        string name = fm.Groups[1].Value;
        var args = new Dictionary<string, object?>();
        foreach (Match m in ParameterRegex().Matches(body, fm.Index + fm.Length))
        {
            string key = m.Groups[1].Value;
            string raw = m.Groups[2].Value;
            if (raw.StartsWith('\n'))
            {
                raw = raw[1..];
            }

            if (raw.EndsWith('\n'))
            {
                raw = raw[..^1];
            }

            args[key] = Coerce(raw, tools.ParameterType(name, key));
        }
        return new FunctionCallContent(ToolCallIds.New(), name, args);
    }

    private static object? Coerce(string raw, string? type)
    {
        if (type == "string")
        {
            return raw;
        }

        var t = raw.Trim();
        if (t.Length > 0 && (type is null or "integer" or "number" or "boolean" or "object" or "array" or "null"))
        {
            try
            {
                using var doc = JsonDocument.Parse(t);
                return doc.RootElement.Clone();
            }
            catch (JsonException) { }
        }
        return raw;
    }
}

internal static class ToolCallIds
{
    public static string New() => "call_" + Guid.NewGuid().ToString("N")[..24];
}
