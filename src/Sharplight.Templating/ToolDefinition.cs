namespace Sharplight.Templating;

/// <summary>A function tool offered to the model. <see cref="Parameters"/> is a JSON-schema object.</summary>
public sealed record ToolDefinition(string Name, string? Description = null, IReadOnlyDictionary<string, object?>? Parameters = null)
{
    internal JinjaDict ToValue()
    {
        var fn = new JinjaDict { ["name"] = Name };
        if (Description != null)
        {
            fn["description"] = Description;
        }

        if (Parameters != null)
        {
            fn["parameters"] = JinjaValue.From(Parameters);
        }

        return new JinjaDict { ["type"] = "function", ["function"] = fn };
    }

    /// <summary>
    /// Parses an OpenAI-style tool object (<c>{"type":"function","function":{"name":..,"description":..,"parameters":{..}}}</c>)
    /// or the bare function object. Property order is preserved.
    /// </summary>
    public static ToolDefinition FromJson(string json)
    {
        if (JinjaJson.Parse(json) is not JinjaDict root)
        {
            throw new ArgumentException("Tool definition must be a JSON object", nameof(json));
        }

        if (root.TryGetValue("function", out var f) && f is JinjaDict fd)
        {
            root = fd;
        }

        string name = root.TryGetValue("name", out var n) && n is string ns ? ns : throw new ArgumentException("Tool definition has no name", nameof(json));
        string? desc = root.TryGetValue("description", out var d) ? d as string : null;
        IReadOnlyDictionary<string, object?>? pars = null;
        if (root.TryGetValue("parameters", out var p) && p is JinjaDict pd)
        {
            pars = pd.ToDictionary(kv => (string)kv.Key, kv => kv.Value);
        }

        return new ToolDefinition(name, desc, pars);
    }
}
