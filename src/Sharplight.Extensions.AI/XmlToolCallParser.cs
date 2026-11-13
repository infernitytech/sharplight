using System.Text.Json;
using System.Text.RegularExpressions;
using Microsoft.Extensions.AI;

namespace Sharplight.Extensions.AI;

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
