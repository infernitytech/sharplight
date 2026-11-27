using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Salesforce Moirai-MoE (the uni2ts package's <c>MoiraiMoEModule</c>: Salesforce/moirai-moe-1.0-R-small / -base) → canonical <c>moirai_moe</c>
/// (documented on <c>MoiraiAdapter</c>). The config has the keys of Moirai 1.x; <see cref="MoiraiHfMapping.Detect"/> tells them apart by the routing
/// centroids in the weights. The multi-size layers (<c>in_proj</c>, <c>res_proj</c>, <c>feat_proj</c>) and the distribution heads are split per patch size as
/// for Moirai 1.x; the 32 experts of every layer's feed-forward block are stacked into <c>encoder.layers.{i}.ffn.experts.{gate,up,down}.weight</c>
/// (from <c>fc_gate</c>, <c>fc1</c>, <c>fc2</c>) and the layer's centroids (<c>encoder.centroid</c> [layers, 32, d], f64) become
/// <c>encoder.layers.{i}.ffn.centroid</c> [32, d] in f32, the type the module casts them to.
/// </summary>
internal sealed partial class MoiraiMoeHfMapping : IHfArchitectureMapping
{
    /// <summary>The model type of Moirai-MoE.</summary>
    public const string ModelType = "moirai_moe";

    /// <inheritdoc/>
    public string Architecture => ModelType;

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = [ModelType];

    /// <inheritdoc/>
    public bool NeedsTokenizer => false;

    /// <inheritdoc/>
    public IEnumerable<string> Validate(HfConfig c)
    {
        if (!MoiraiHfMapping.TryComponents(c, out _, out string? error))
        {
            yield return error!.Replace("moirai:", "moirai_moe:", StringComparison.Ordinal);
        }

        if (c.GetBool("scaling") == false)
        {
            yield return "moirai_moe: checkpoints without scaling (scaling = false) are not supported.";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HuggingFaceImportOptions options)
    {
        string a = Architecture + ".";
        MoiraiHfMapping.TryComponents(c, out var components, out _);
        int[] sizes = MoiraiHfMapping.PatchSizes(c);
        int hidden = c.Int("d_model");
        return new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "hidden_size"] = hidden,
            [a + "intermediate_size"] = c.GetInt("d_ff") ?? ((((4 * hidden) * 2 / 3) + 7) / 8 * 8),
            [a + "layer_count"] = c.Int("num_layers"),
            [a + "patch_sizes"] = string.Join(',', sizes),
            [a + "patch_size"] = sizes.Contains(16) ? 16 : sizes[0],
            [a + "components"] = string.Join(',', components!.Select(x => x.Config)),
            [a + "max_sequence_length"] = c.GetInt("max_seq_len") ?? 512,
            [a + "expert_count"] = 32,
            [a + "expert_used_count"] = 2,
        };
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
    {
        MoiraiHfMapping.TryComponents(c, out var components, out _);
        int[] sizes = MoiraiHfMapping.PatchSizes(c);
        int layers = c.Int("num_layers");
        var collector = new HfWeightCollector();
        var heads = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
        var multi = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
        var experts = new SortedDictionary<(int Layer, string Kind), SortedDictionary<int, WeightInfo>>();
        WeightInfo? centroids = null;
        foreach (var (name, tensor) in tensors)
        {
            var info = tensor.ToWeightInfo();
            if (ExpertName().Match(name) is { Success: true } m)
            {
                string kind = m.Groups[3].Value switch
                {
                    "fc_gate" => "gate",
                    "fc1" => "up",
                    _ => "down",
                };
                var key = (int.Parse(m.Groups[1].Value), kind);
                if (!experts.TryGetValue(key, out var byIndex))
                {
                    experts[key] = byIndex = [];
                }

                byIndex[int.Parse(m.Groups[2].Value)] = info;
            }
            else if (name == "encoder.centroid")
            {
                centroids = info;
            }
            else if (name.StartsWith("encoder.", StringComparison.Ordinal))
            {
                collector.Add(HfTransforms.Plain(info, name));
            }
            else if (name is "in_proj.weight" or "in_proj.bias" or "res_proj.weight" or "res_proj.bias" or "feat_proj.weight" or "feat_proj.bias")
            {
                multi[name] = info;
            }
            else if (name.StartsWith("param_proj.proj.", StringComparison.Ordinal))
            {
                heads[name["param_proj.proj.".Length..]] = info;
            }
            else
            {
                collector.AddUnmapped(name);
            }
        }

        collector.ThrowIfUnmapped(ModelType);
        foreach (string required in (string[])["in_proj.weight", "in_proj.bias", "res_proj.weight", "res_proj.bias", "feat_proj.weight", "feat_proj.bias"])
        {
            if (!multi.ContainsKey(required))
            {
                throw new InvalidDataException($"moirai_moe: the checkpoint has no {required}.");
            }
        }

        if (centroids is null || centroids.Shape.Rank != 3 || centroids.Shape[0].Fixed != layers)
        {
            throw new InvalidDataException("moirai_moe: the checkpoint has no encoder.centroid [layers, experts, d].");
        }

        foreach (var ((layer, kind), byIndex) in experts)
        {
            if (byIndex.Count != 32 || byIndex.Keys.Max() != 31)
            {
                throw new InvalidDataException($"moirai_moe: layer {layer}: expected 32 {kind} experts, found {byIndex.Count}.");
            }

            collector.Add(WeightInfo.Stack($"encoder.layers.{layer}.ffn.experts.{kind}.weight", [.. byIndex.Values]));
        }

        int hidden = checked((int)multi["in_proj.weight"].Shape[1].Fixed);
        int count = checked((int)centroids.Shape[1].Fixed);
        for (int layer = 0; layer < layers; layer++)
        {
            int index = layer;
            collector.Add(WeightInfo.Derived($"encoder.layers.{layer}.ffn.centroid", DataType.F32, new Shape(count, hidden), () =>
                HfTransforms.EncodeF32(HfTransforms.DecodeF32(centroids)[(index * count * hidden)..((index + 1) * count * hidden)])));
        }

        int maxSize = sizes.Max();
        MoiraiHfMapping.AddPerSize(collector, "in_proj", multi["in_proj.weight"], multi["in_proj.bias"], sizes, hidden, maxSize, p => p);
        MoiraiHfMapping.AddPerSize(collector, "res_proj", multi["res_proj.weight"], multi["res_proj.bias"], sizes, hidden, maxSize, p => p);
        MoiraiHfMapping.AddPerSize(collector, "feat_proj", multi["feat_proj.weight"], multi["feat_proj.bias"], sizes, hidden, hidden, _ => hidden);
        MoiraiHfMapping.AddParamProj(collector, heads, components!, sizes, hidden);
        return collector.Weights;
    }

    [GeneratedRegex(@"^encoder\.layers\.(\d+)\.ffn\.experts\.(\d+)\.(fc_gate|fc1|fc2)\.weight$")]
    private static partial Regex ExpertName();
}
