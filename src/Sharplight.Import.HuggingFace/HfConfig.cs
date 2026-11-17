using System.Globalization;
using System.Text.Json;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// A checkpoint's <c>config.json</c>. Like llama.cpp's converter, the nested <c>text_config</c> (if any; else <c>llm_config</c>, as
/// SenseNova-U1 names it) is merged over the root object (the nested config wins), so hyper-parameters are looked up flat.
/// </summary>
internal sealed partial class HfConfig : IDisposable
{
    private readonly JsonDocument _doc;
    private readonly Dictionary<string, JsonElement> _flat = new(StringComparer.Ordinal);
    private readonly string? _modelType;     // a view's model type (AsModelType); null: from the document
    private readonly bool _ownsDocument = true;

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

        // Qwen's omni-style checkpoints (Qwen3-ASR) nest the language model one level deeper: thinker_config.text_config.
        bool nested = doc.RootElement.TryGetProperty("thinker_config", out var thinker) && thinker.ValueKind == JsonValueKind.Object;
        var outer = nested ? thinker : doc.RootElement;
        if (nested)
        {
            foreach (var p in thinker.EnumerateObject())
            {
                if (p.Name != "model_type")
                {
                    _flat[p.Name] = p.Value;
                }
            }
        }

        if ((outer.TryGetProperty("text_config", out var tc) || outer.TryGetProperty("llm_config", out tc)) && tc.ValueKind == JsonValueKind.Object)
        {
            foreach (var p in tc.EnumerateObject())
            {
                _flat[p.Name] = p.Value;
            }
        }
    }

    /// <summary>
    /// A quantization object of the checkpoint (<c>quantization_config</c>, or MLX's <c>quantization</c>): the root's, else the nested
    /// <c>text_config</c> / <c>llm_config</c>'s (Kimi-K3 keeps its compressed-tensors config there). Unlike the merged lookups, the root
    /// wins: it describes the whole checkpoint.
    /// </summary>
    /// <param name="name">The property name.</param>
    /// <param name="settings">The object when found.</param>
    /// <returns>Whether an object of that name exists.</returns>
    public bool TryGetQuantization(string name, out JsonElement settings)
    {
        if (Root.TryGetProperty(name, out settings) && settings.ValueKind == JsonValueKind.Object)
        {
            return true;
        }

        foreach (string nested in (string[])["text_config", "llm_config"])
        {
            if (Root.TryGetProperty(nested, out var inner) && inner.ValueKind == JsonValueKind.Object
                && inner.TryGetProperty(name, out settings) && settings.ValueKind == JsonValueKind.Object)
            {
                return true;
            }
        }

        settings = default;
        return false;
    }

    [System.Text.RegularExpressions.GeneratedRegex(@"(?<=[:\[,]\s*)(-?Infinity|NaN)")]
    private static partial System.Text.RegularExpressions.Regex NonFiniteLiteral();

    /// <summary>Loads <c>config.json</c> from <paramref name="directory"/>; trailing commas are tolerated.</summary>
    public static HfConfig Load(string directory)
    {
        return Parse(File.ReadAllText(Path.Combine(directory, "config.json")), directory);
    }

    /// <summary>Parses the text of a <c>config.json</c>; trailing commas are tolerated.</summary>
    /// <param name="text">The file's text.</param>
    /// <param name="directory">The checkpoint directory, or empty when the config was read without one (e.g. from the Hub).</param>
    /// <exception cref="JsonException">The text is not JSON.</exception>
    public static HfConfig Parse(string text, string directory)
    {
        // Python's json module writes non-finite numbers as bare Infinity / NaN (e.g. Mamba-2 time_step_limit), which JSON proper lacks.
        text = NonFiniteLiteral().Replace(text, m => m.Value switch { "-Infinity" => "-1e308", "Infinity" => "1e308", _ => "null" });
        var document = JsonDocument.Parse(text, new JsonDocumentOptions { AllowTrailingCommas = true });
        return new HfConfig(document, directory);
    }

    /// <summary>
    /// The model type: the root <c>model_type</c> (e.g. "qwen3_5"), falling back to text_config's, then to the model type of a
    /// remote-code <c>architectures</c> class whose configs omit it (e.g. <c>BailingMoeV2ForCausalLM</c>). EAGLE-3 drafters, whose configs say
    /// <c>llama</c>, are <c>eagle3</c>; remote-code classes whose configs name another family (<c>GroveMoeForCausalLM</c>, configs say
    /// <c>qwen3_moe</c>) get their own type.
    /// </summary>
    public string? ModelType
        => _modelType ?? (Eagle3HfMapping.IsDrafter(this) ? "eagle3"
            : Architectures.Select(a => s_remoteCodeOverrides.GetValueOrDefault(a)).FirstOrDefault(t => t is not null) is { } overridden ? overridden
            : Root.TryGetProperty("model_type", out var m) && m.ValueKind == JsonValueKind.String ? m.GetString()
            : GetString("model_type") ?? Architectures.Select(a => s_remoteCodeModelTypes.GetValueOrDefault(a)).FirstOrDefault(t => t is not null)
                ?? (Root.TryGetProperty("sources", out _) && Root.TryGetProperty("t_layers", out _) ? "htdemucs" : TimmModelType()));

    /// <summary>The model type of timm ResNet checkpoints (<c>timm/resnet50.a1_in1k</c>: an <c>architecture</c> name and a <c>pretrained_cfg</c> instead of a <c>model_type</c>).</summary>
    public const string TimmResNetModelType = "timm_resnet";

    /// <summary>The model type of timm ConvNeXt checkpoints (<c>timm/convnext_tiny.fb_in1k</c>).</summary>
    public const string TimmConvNextModelType = "timm_convnext";

    private string? TimmModelType()
    {
        if (Root.ValueKind != JsonValueKind.Object || !Root.TryGetProperty("pretrained_cfg", out _) || !Root.TryGetProperty("architecture", out var a) || a.ValueKind != JsonValueKind.String)
        {
            return null;
        }

        string name = a.GetString()!;
        if (name.StartsWith("resnet", StringComparison.Ordinal) || name.StartsWith("wide_resnet", StringComparison.Ordinal))
        {
            return TimmResNetModelType;
        }

        return name.StartsWith("convnext_", StringComparison.Ordinal) ? TimmConvNextModelType : null;
    }

    private HfConfig(HfConfig config, string modelType)
    {
        _doc = config._doc;
        _flat = config._flat;
        Directory = config.Directory;
        _modelType = modelType;
        _ownsDocument = false;
    }

    /// <summary>
    /// A view of this config that reports <paramref name="modelType"/> as its <see cref="ModelType"/>: a drafter whose layers are another model type's
    /// (a Gemma 4 assistant's are <c>gemma4_text</c> layers) maps them with that type's mapping. The view shares this config's document; disposing it does
    /// nothing.
    /// </summary>
    public HfConfig AsModelType(string modelType) => new(this, modelType);

    // Remote-code causal-LM classes whose config.json may lack model_type.
    private static readonly Dictionary<string, string> s_remoteCodeModelTypes = new(StringComparer.Ordinal)
    {
        ["BailingMoeV2ForCausalLM"] = "bailing_moe_v2",
        ["CogVLMForCausalLM"] = "cogvlm",
        ["BiRefNet"] = "birefnet",
    };

    // Remote-code causal-LM classes whose config.json names another family's model_type (GroveMoE says qwen3_moe): the class decides.
    private static readonly Dictionary<string, string> s_remoteCodeOverrides = new(StringComparer.Ordinal)
    {
        ["GroveMoeForCausalLM"] = "grove_moe",
        ["modeling_grove_moe.GroveMoeForCausalLM"] = "grove_moe",
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
    public void Dispose()
    {
        if (_ownsDocument)
        {
            _doc.Dispose();
        }
    }
}
