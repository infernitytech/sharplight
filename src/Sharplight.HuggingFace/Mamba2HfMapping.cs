using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>Hugging Face <c>mamba2</c> (pure Mamba-2 stack: <c>backbone.layers.N.{norm,mixer}</c>) → canonical <c>hybrid</c> (documented on <c>HybridAdapter</c>).</summary>
internal sealed partial class Mamba2HfMapping : IHfArchitectureMapping
{
    /// <inheritdoc/>
    public string Architecture => "hybrid";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["mamba2"];

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HfImportOptions options)
    {
        string a = Architecture + ".";
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "layer_count"] = c.Int("num_hidden_layers"),
            [a + "hidden_size"] = c.Int("hidden_size"),
            [a + "norm_eps"] = (float)(c.GetDouble("layer_norm_epsilon", "rms_norm_eps") ?? 1e-5),
            [a + "ssm.conv_kernel"] = c.Int("conv_kernel"),
            [a + "ssm.state_size"] = c.Int("state_size"),
            [a + "ssm.group_count"] = c.Int("n_groups"),
            [a + "ssm.head_count"] = c.Int("num_heads"),
            [a + "ssm.head_dim"] = c.Int("head_dim"),
        };
        HfMappingHelpers.AddContextLength(md, a, c);
        return md;
    }

    [GeneratedRegex(@"^backbone\.layers\.(\d+)\.(.+)$")]
    private static partial Regex LayerName();

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HfImportOptions options)
    {
        var collector = new HfWeightCollector();
        foreach (var (name, tensor) in tensors.OrderBy(p => p.Key, StringComparer.Ordinal))
        {
            var w = tensor.ToWeightInfo();
            var m = LayerName().Match(name);
            bool mapped = m.Success ? TryMapLayerTensor($"layers.{m.Groups[1].Value}.", m.Groups[2].Value, w, collector) : TryMapGlobalTensor(c, name, w, collector);
            if (!mapped)
            {
                collector.AddUnmapped(name);
            }
        }

        collector.ThrowIfUnmapped("mamba2");
        return collector.Weights;
    }

    private static bool TryMapGlobalTensor(HfConfig c, string name, WeightInfo w, HfWeightCollector collector)
    {
        switch (name)
        {
            case "backbone.embeddings.weight":
                collector.Add(HfTransforms.Plain(w, "token_embedding.weight"));
                return true;
            case "backbone.norm_f.weight":
                collector.Add(HfTransforms.Plain(w, "output_norm.weight"));
                return true;
            case "lm_head.weight":
                if (!(c.GetBool("tie_word_embeddings") ?? false))
                {
                    collector.Add(HfTransforms.Plain(w, "lm_head.weight"));
                }

                return true;
            default:
                return false;
        }
    }

    private static bool TryMapLayerTensor(string prefix, string suffix, WeightInfo w, HfWeightCollector collector)
    {
        if (suffix == "norm.weight")
        {
            collector.Add(HfTransforms.Plain(w, prefix + "input_norm.weight"));
            return true;
        }

        return suffix.StartsWith("mixer.", StringComparison.Ordinal) && HybridHfTensors.TryMapSsm(prefix, suffix["mixer.".Length..], w, collector);
    }
}
