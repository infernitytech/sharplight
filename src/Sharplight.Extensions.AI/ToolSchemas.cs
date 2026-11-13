using System.Text.Json;
using Microsoft.Extensions.AI;

namespace Sharplight.Extensions.AI;

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
