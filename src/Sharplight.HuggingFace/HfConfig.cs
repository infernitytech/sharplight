using System.Globalization;
using System.Text.Json;

namespace Sharplight.HuggingFace;

/// <summary>
/// A checkpoint's <c>config.json</c>. Like llama.cpp's converter, the nested <c>text_config</c> (if any)
/// is merged over the root object (text_config wins), so hyper-parameters are looked up flat.
/// </summary>
internal sealed partial class HfConfig : IDisposable
{
    private readonly JsonDocument _doc;
    private readonly Dictionary<string, JsonElement> _flat = new(StringComparer.Ordinal);

    /// <summary>The parsed root object of <c>config.json</c> (before merging <c>text_config</c>).</summary>
    public JsonElement Root => _doc.RootElement;
    /// <summary>The checkpoint directory the config was loaded from.</summary>
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

    [System.Text.RegularExpressions.GeneratedRegex(@"(?<=[:\[,]\s*)(-?Infinity|NaN)")]
    private static partial System.Text.RegularExpressions.Regex NonFiniteLiteral();

    /// <summary>Loads <c>config.json</c> from <paramref name="directory"/>; trailing commas are tolerated.</summary>
    public static HfConfig Load(string directory)
    {
        var path = Path.Combine(directory, "config.json");
        // Python's json module writes non-finite numbers as bare Infinity / NaN (e.g. Mamba-2 time_step_limit), which JSON proper lacks.
        string text = NonFiniteLiteral().Replace(File.ReadAllText(path), m => m.Value switch { "-Infinity" => "-1e308", "Infinity" => "1e308", _ => "null" });
        var document = JsonDocument.Parse(text, new JsonDocumentOptions { AllowTrailingCommas = true });
        return new HfConfig(document, directory);
    }

    /// <summary>
    /// The model type: the root <c>model_type</c> (e.g. "qwen3_5"), falling back to text_config's, then to the model type of a
    /// remote-code <c>architectures</c> class whose configs omit it (e.g. <c>BailingMoeV2ForCausalLM</c>). EAGLE-3 drafters, whose configs say
    /// <c>llama</c>, are <c>eagle3</c>.
    /// </summary>
    public string? ModelType
        => Eagle3HfMapping.IsDrafter(this) ? "eagle3"
            : Root.TryGetProperty("model_type", out var m) && m.ValueKind == JsonValueKind.String ? m.GetString()
            : GetString("model_type") ?? Architectures.Select(a => s_remoteCodeModelTypes.GetValueOrDefault(a)).FirstOrDefault(t => t is not null);

    // Remote-code causal-LM classes whose config.json may lack model_type.
    private static readonly Dictionary<string, string> s_remoteCodeModelTypes = new(StringComparer.Ordinal)
    {
        ["BailingMoeV2ForCausalLM"] = "bailing_moe_v2",
    };

    /// <summary>The <c>architectures</c> class names (e.g. "Qwen3ForCausalLM"); empty when absent.</summary>
    public IReadOnlyList<string> Architectures
        => Root.TryGetProperty("architectures", out var a) && a.ValueKind == JsonValueKind.Array
            ? a.EnumerateArray().Select(x => x.GetString() ?? "").ToArray() : [];

    /// <summary>Whether the flat config has a non-null value for <paramref name="key"/>.</summary>
    public bool Has(string key) => _flat.TryGetValue(key, out var v) && v.ValueKind != JsonValueKind.Null;

    /// <summary>The first non-null value among <paramref name="keys"/> (aliases used by different checkpoint generations), or null.</summary>
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

    /// <summary>The first present key as an integer (numeric strings and booleans convert), or null.</summary>
    public int? GetInt(params string[] keys) => Get(keys) is { } v ? (int)ToDouble(v) : null;
    /// <summary>The first present key as a double, or null.</summary>
    public double? GetDouble(params string[] keys) => Get(keys) is { } v ? ToDouble(v) : null;
    /// <summary>The first present key as a boolean, or null.</summary>
    public bool? GetBool(params string[] keys) => Get(keys) is { } v ? v.ValueKind == JsonValueKind.True : null;
    /// <summary>The first present key as a string; null when absent or not a string.</summary>
    public string? GetString(params string[] keys) => Get(keys) is { ValueKind: JsonValueKind.String } v ? v.GetString() : null;

    /// <summary>Like <see cref="GetInt"/> but throws <see cref="KeyNotFoundException"/> when no key is present.</summary>
    public int Int(params string[] keys) => GetInt(keys) ?? throw Missing(keys);
    /// <summary>Like <see cref="GetDouble"/> but throws <see cref="KeyNotFoundException"/> when no key is present.</summary>
    public double Double(params string[] keys) => GetDouble(keys) ?? throw Missing(keys);

    /// <summary>The (merged) <c>rope_parameters</c> / legacy <c>rope_scaling</c> object, or null.</summary>
    public JsonElement? RopeParameters
        => Get("rope_parameters", "rope_scaling") is { ValueKind: JsonValueKind.Object } v ? v : null;

    /// <summary>
    /// The rope parameters of the global-attention layers: <see cref="RopeParameters"/>, or its <c>full_attention</c> entry when the config
    /// keeps one object per layer type (Gemma 3, ModernBERT as transformers 5 saves them; <see cref="LayerTypeRopeDouble"/> reads the others).
    /// </summary>
    public JsonElement? GlobalRopeParameters
        => RopeParameters is { } v && v.TryGetProperty("full_attention", out var full) && full.ValueKind == JsonValueKind.Object ? full : RopeParameters;

    /// <summary>A RoPE parameter of one layer type (e.g. <c>sliding_attention</c>) in a per-layer-type <c>rope_parameters</c> object, or null.</summary>
    public double? LayerTypeRopeDouble(string layerType, string key)
        => Get("rope_parameters") is { ValueKind: JsonValueKind.Object } parameters && parameters.TryGetProperty(layerType, out var type)
           && type.ValueKind == JsonValueKind.Object && type.TryGetProperty(key, out var value) && value.ValueKind == JsonValueKind.Number
            ? value.GetDouble()
            : null;

    /// <summary>
    /// Looks up a RoPE parameter in the global-attention <c>rope_parameters</c> (<see cref="GlobalRopeParameters"/>) first, then at the config's top
    /// level (older checkpoints keep <c>rope_theta</c> / <c>partial_rotary_factor</c> there).
    /// </summary>
    public double? RopeDouble(string rope, params string[] topLevel)
    {
        if (GlobalRopeParameters is { } rp && rp.TryGetProperty(rope, out var v) && v.ValueKind == JsonValueKind.Number)
        {
            return v.GetDouble();
        }

        return GetDouble(topLevel);
    }

    /// <summary>An integer array from the global-attention <c>rope_parameters</c> (e.g. <c>mrope_section</c>), or null.</summary>
    public int[]? RopeIntArray(string rope)
        => GlobalRopeParameters is { } rp && rp.TryGetProperty(rope, out var v) && v.ValueKind == JsonValueKind.Array
            ? v.EnumerateArray().Select(x => x.GetInt32()).ToArray() : null;

    /// <summary>A string-array value, or null when absent or not an array.</summary>
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

    /// <summary>Releases the parsed document; elements obtained from it must not be used afterwards.</summary>
    public void Dispose() => _doc.Dispose();
}
