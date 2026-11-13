using System.Text.Json;
using Microsoft.Extensions.AI;

namespace Sharplight.Extensions.AI;

/// <summary>Hermes / Qwen2.5 / Qwen3 style: <c>{"name": "f", "arguments": {...}}</c> inside the tool-call tags.</summary>
public sealed class JsonToolCallParser : IToolCallParser
{
    /// <inheritdoc/>
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
        catch (JsonException)
        {
            return null;
        }
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
