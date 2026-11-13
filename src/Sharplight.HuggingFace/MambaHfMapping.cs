using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.HuggingFace;

/// <summary>Hugging Face <c>mamba</c> (Mamba-1: <c>backbone.layers.N.{norm,mixer}</c>) → canonical <c>hybrid</c> with Mamba-1 layers (documented on <c>HybridAdapter</c> and <c>Mamba1Layer</c>).</summary>
internal sealed partial class MambaHfMapping : IHfArchitectureMapping
{
    /// <inheritdoc/>
    public string Architecture => "hybrid";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["mamba"];

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
        };
        HfMappingHelpers.AddContextLength(md, a, c);
        return md;
    }

    [GeneratedRegex(@"^backbone\.layers\.(\d+)\.(.+)$")]
    private static partial Regex LayerName();

    /// <summary>Per-layer tensors that are only renamed, HF suffix to canonical suffix.</summary>
    private static readonly Dictionary<string, string> s_layer = new(StringComparer.Ordinal)
    {
        ["norm.weight"] = "input_norm.weight",
        ["mixer.in_proj.weight"] = "ssm.in.weight",
        ["mixer.out_proj.weight"] = "ssm.out.weight",
        ["mixer.x_proj.weight"] = "ssm.x.weight",
        ["mixer.dt_proj.weight"] = "ssm.dt.weight",
    };

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

        collector.ThrowIfUnmapped("mamba");
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
                if (!(c.GetBool("tie_word_embeddings") ?? true))
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
        if (s_layer.TryGetValue(suffix, out var renamed))
        {
            collector.Add(HfTransforms.Plain(w, prefix + renamed));
            return true;
        }

        switch (suffix)
        {
            case "mixer.A_log":
                collector.Add(HfTransforms.ToF32(w, prefix + "ssm.a", map: static v => -MathF.Exp(v)));   // A = -exp(A_log), stays [inner, N]
                return true;
            case "mixer.D":
                collector.Add(HfTransforms.ToF32(w, prefix + "ssm.d"));
                return true;
            case "mixer.dt_proj.bias":
                collector.Add(HfTransforms.ToF32(w, prefix + "ssm.dt_bias"));
                return true;
            case "mixer.conv1d.bias":
                collector.Add(HfTransforms.ToF32(w, prefix + "ssm.conv1d.bias"));
                return true;
            case "mixer.conv1d.weight":
                collector.Add(HfTransforms.ToF32(w, prefix + "ssm.conv1d.weight", new Shape(w.Shape[0].Fixed, w.Shape[w.Shape.Rank - 1].Fixed)));
                return true;
            default:
                return false;
        }
    }
}
