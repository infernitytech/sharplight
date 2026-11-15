using System.Text.Json;
using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>
/// vllm-sr Decision 2.0 packages (<c>config.json</c> with <c>"decision_format": "vllm-sr-decision"</c>): the backbone checkpoint lives in a
/// subdirectory (<c>backbone.config</c>, <c>backbone.weights</c>) and the "endpoint" span scoring head in <c>decision_head.safetensors</c>
/// (<c>decision_weights.decision_head</c>). The backbone is imported with its own mapping; the head's tensors become <c>span_head.*</c>.
/// </summary>
internal sealed class HfDecisionPackage
{
    private const string Format = "vllm-sr-decision";

    private static readonly Dictionary<string, string> s_head = new(StringComparer.Ordinal)
    {
        ["candidate_norm.weight"] = "span_head.option_norm.weight",
        ["candidate_norm.bias"] = "span_head.option_norm.bias",
        ["query_norm.weight"] = "span_head.question_norm.weight",
        ["query_norm.bias"] = "span_head.question_norm.bias",
        ["key.weight"] = "span_head.key.weight",
        ["query.weight"] = "span_head.query.weight",
        ["candidate_mlp.weight"] = "span_head.option_mlp.weight",
        ["candidate_mlp.bias"] = "span_head.option_mlp.bias",
        ["query_mlp.weight"] = "span_head.question_mlp.weight",
        ["scalar.weight"] = "span_head.scorer.weight",
    };

    /// <summary>The backbone checkpoint directory (its <c>config.json</c> and safetensors shards).</summary>
    public required string BackboneDirectory { get; init; }

    /// <summary>The head's safetensors file.</summary>
    public required string HeadPath { get; init; }

    /// <summary>Reads the package description of <paramref name="dir"/>; null when it is not a Decision 2.0 package.</summary>
    /// <exception cref="InvalidDataException">The package's backbone or head is missing.</exception>
    public static HfDecisionPackage? Find(string dir)
    {
        string path = Path.Combine(dir, "config.json");
        if (!File.Exists(path))
        {
            return null;
        }

        // The config parser tolerates what Python's json module writes (bare Infinity / NaN, trailing commas).
        using var config = HfConfig.Load(dir);
        var root = config.Root;
        if (root.ValueKind != JsonValueKind.Object || !root.TryGetProperty("decision_format", out var format) || format.GetString() != Format)
        {
            return null;
        }

        string backboneConfig = root.TryGetProperty("backbone", out var backbone) && backbone.TryGetProperty("config", out var c)
            ? c.GetString() ?? "backbone/config.json" : "backbone/config.json";
        string head = root.TryGetProperty("decision_weights", out var weights) && weights.TryGetProperty("decision_head", out var h)
            ? h.GetString() ?? "decision_head.safetensors" : "decision_head.safetensors";
        var package = new HfDecisionPackage
        {
            BackboneDirectory = Path.GetDirectoryName(Path.GetFullPath(Path.Combine(dir, backboneConfig)))!,
            HeadPath = Path.GetFullPath(Path.Combine(dir, head)),
        };
        if (!File.Exists(Path.Combine(package.BackboneDirectory, "config.json")) || !File.Exists(package.HeadPath))
        {
            throw new InvalidDataException($"{dir}: Decision 2.0 package without its backbone config ({backboneConfig}) or head ({head}).");
        }

        return package;
    }

    /// <summary>Adds the head's hyper-parameters (type "endpoint", LayerNorm epsilon 1e-5 as in torch) to <paramref name="md"/>.</summary>
    public static void AddHyperParameters(IDictionary<string, object> md)
    {
        md[CanonicalKeys.SpanHeadType] = "endpoint";
        md[CanonicalKeys.SpanHeadNormEps] = 1e-5f;
    }

    /// <summary>Opens the head file (added to <paramref name="owned"/>) and returns its tensors under canonical names.</summary>
    /// <exception cref="NotSupportedException">The head has a tensor this importer does not know.</exception>
    public IReadOnlyDictionary<string, WeightInfo> HeadWeights(List<IDisposable> owned)
    {
        var file = SafetensorsFile.Open(HeadPath);
        owned.Add(file);
        var weights = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
        foreach (var (name, tensor) in file.Tensors)
        {
            string canonical = s_head.GetValueOrDefault(name) ?? throw new NotSupportedException($"{HeadPath}: unexpected decision head tensor '{name}'.");
            weights[canonical] = HfTransforms.Plain(tensor.ToWeightInfo(), canonical);
        }

        return weights;
    }
}
