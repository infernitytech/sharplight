using System.Globalization;
using System.Text.Json;

namespace Sharplight.HuggingFace;

/// <summary>
/// A checkpoint's <c>config.json</c>. Like llama.cpp's converter, the nested <c>text_config</c> (if any)
/// is merged over the root object (text_config wins), so hyper-parameters are looked up flat.
/// </summary>
public sealed class HfConfig : IDisposable
{
    private readonly JsonDocument _doc;
    private readonly Dictionary<string, JsonElement> _flat = new(StringComparer.Ordinal);

    public JsonElement Root => _doc.RootElement;
    public string Directory { get; }

    private HfConfig(JsonDocument doc, string directory)
    {
        _doc = doc;
        Directory = directory;
        foreach (var p in doc.RootElement.EnumerateObject())
        {
            _flat[p.Name] = p.Value;
        }

        if (doc.RootElement.TryGetProperty("text_config", out var tc) && tc.ValueKind == JsonValueKind.Object)
        {
            foreach (var p in tc.EnumerateObject())
            {
                _flat[p.Name] = p.Value;
            }
        }
    }

    public static HfConfig Load(string directory)
    {
        var path = Path.Combine(directory, "config.json");
        return new HfConfig(JsonDocument.Parse(File.ReadAllBytes(path), new JsonDocumentOptions { AllowTrailingCommas = true }), directory);
    }

    /// <summary>The model type: the root <c>model_type</c> (e.g. "qwen3_5"), falling back to text_config's.</summary>
    public string? ModelType
        => Root.TryGetProperty("model_type", out var m) && m.ValueKind == JsonValueKind.String ? m.GetString() : GetString("model_type");

    public IReadOnlyList<string> Architectures
        => Root.TryGetProperty("architectures", out var a) && a.ValueKind == JsonValueKind.Array
            ? a.EnumerateArray().Select(x => x.GetString() ?? "").ToArray() : [];

    public bool Has(string key) => _flat.TryGetValue(key, out var v) && v.ValueKind != JsonValueKind.Null;

    public JsonElement? Get(params string[] keys)
    {
        foreach (var k in keys)
        {
            if (_flat.TryGetValue(k, out var v) && v.ValueKind != JsonValueKind.Null)
            {
                return v;
            }
        }

        return null;
    }

    public int? GetInt(params string[] keys) => Get(keys) is { } v ? (int)ToDouble(v) : null;
    public double? GetDouble(params string[] keys) => Get(keys) is { } v ? ToDouble(v) : null;
    public bool? GetBool(params string[] keys) => Get(keys) is { } v ? v.ValueKind == JsonValueKind.True : null;
    public string? GetString(params string[] keys) => Get(keys) is { ValueKind: JsonValueKind.String } v ? v.GetString() : null;

    public int Int(params string[] keys) => GetInt(keys) ?? throw Missing(keys);
    public double Double(params string[] keys) => GetDouble(keys) ?? throw Missing(keys);

    /// <summary>The (merged) <c>rope_parameters</c> / legacy <c>rope_scaling</c> object, or null.</summary>
    public JsonElement? RopeParameters
        => Get("rope_parameters", "rope_scaling") is { ValueKind: JsonValueKind.Object } v ? v : null;

    /// <summary>
    /// Looks up a RoPE parameter in <c>rope_parameters</c> first, then at the config's top level
    /// (older checkpoints keep <c>rope_theta</c> / <c>partial_rotary_factor</c> there).
    /// </summary>
    public double? RopeDouble(string rope, params string[] topLevel)
    {
        if (RopeParameters is { } rp && rp.TryGetProperty(rope, out var v) && v.ValueKind == JsonValueKind.Number)
        {
            return v.GetDouble();
        }

        return GetDouble(topLevel);
    }

    public int[]? RopeIntArray(string rope)
        => RopeParameters is { } rp && rp.TryGetProperty(rope, out var v) && v.ValueKind == JsonValueKind.Array
            ? v.EnumerateArray().Select(x => x.GetInt32()).ToArray() : null;

    public string[]? GetStringArray(string key)
        => Get(key) is { ValueKind: JsonValueKind.Array } v ? v.EnumerateArray().Select(x => x.GetString() ?? "").ToArray() : null;

    private static double ToDouble(JsonElement v) => v.ValueKind switch
    {
        JsonValueKind.Number => v.GetDouble(),
        JsonValueKind.String => double.Parse(v.GetString()!, CultureInfo.InvariantCulture),
        JsonValueKind.True => 1,
        JsonValueKind.False => 0,
        _ => throw new InvalidDataException($"config.json value of kind {v.ValueKind} is not numeric."),
    };

    private static KeyNotFoundException Missing(string[] keys) => new($"config.json: none of [{string.Join(", ", keys)}] is present.");

    public void Dispose() => _doc.Dispose();
}
