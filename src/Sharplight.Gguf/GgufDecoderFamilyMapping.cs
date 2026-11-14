using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Gguf;

/// <summary>
/// Base of the small per-family mappings of llama.cpp decoder-only architectures that the generic canonical <c>decoder</c> graph
/// expresses: it supplies the shared hyper-parameters (<see cref="GgufDecoderHyperParameters"/>), the shared tensor-name table,
/// fused q/k/v and gate/up splitting and the llama.cpp "normal"-rope un-permutation. A family overrides only what it does differently.
/// </summary>
internal abstract class GgufDecoderFamilyMapping : IGgufArchitectureMapping, IGgufSplitMapping
{
    /// <summary>Tensors outside the layers that only some families have, GGUF name to canonical name.</summary>
    private static readonly Dictionary<string, string> s_extraGlobal = new(StringComparer.Ordinal)
    {
        ["output_norm.bias"] = "output_norm.bias",
        ["output.bias"] = "lm_head.bias",
        ["position_embd.weight"] = "position_embedding.weight",
        ["token_embd_norm.weight"] = "embedding_norm.weight",
        ["token_embd_norm.bias"] = "embedding_norm.bias",
    };

    /// <inheritdoc/>
    public abstract IReadOnlyList<string> GgufArchitectures { get; }

    /// <inheritdoc/>
    public string Architecture => "decoder";

    /// <summary>
    /// Whether llama.cpp's converter stored q/k rows permuted for its interleaved ("normal") rope although the Hugging Face model
    /// rotates halves; the rows are then un-permuted lazily so canonical weights equal those of a Hugging Face import.
    /// </summary>
    protected virtual bool PermutesQk => false;

    /// <summary>Rotary pairing of the canonical rows: "neox" (rotate halves) or "gptj" (interleaved pairs).</summary>
    protected virtual string RopeStyle => "neox";

    /// <summary>Canonical activation name of the MLP.</summary>
    protected virtual string Activation => "silu";

    /// <summary>Whether the model uses mean-centred layer norm instead of RMS norm.</summary>
    protected virtual bool LayerNorm => false;

    /// <summary>Whether the MLP is gated (<c>act(gate) * up</c>); false for plain <c>act(up)</c> blocks.</summary>
    protected virtual bool Gated => true;

    /// <summary>Whether the selected expert weights are renormalized to sum 1 (MoE variants only).</summary>
    protected virtual bool NormalizesExpertWeights => true;

    /// <summary>Family-specific hyper-parameters on top of the shared ones.</summary>
    protected virtual void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
    }

    /// <summary>Adjusts the hyper-parameters once the canonical weights are known (see <see cref="IGgufArchitectureMapping.RefineHyperParameters"/>).</summary>
    protected virtual void RefineFamilyHyperParameters(Dictionary<string, object> md, GgufMetadataView g, IReadOnlyDictionary<string, WeightInfo> weights)
    {
    }

    /// <inheritdoc/>
    public void RefineHyperParameters(Dictionary<string, object> hyperParameters, GgufMetadataView gguf, IReadOnlyDictionary<string, WeightInfo> weights)
        => RefineFamilyHyperParameters(hyperParameters, gguf, weights);

    /// <summary>Tensors outside the layers that only this family has (or names differently), GGUF name to canonical name.</summary>
    protected virtual IReadOnlyDictionary<string, string>? GlobalTensorOverrides => null;

    /// <summary>Per-layer tensors this family names differently from the shared table, GGUF suffix to canonical suffix.</summary>
    protected virtual IReadOnlyDictionary<string, string>? LayerTensorOverrides => null;

    /// <summary>Canonical weights the metadata determines on top of the rope factors (e.g. activation parameters).</summary>
    protected virtual IEnumerable<WeightInfo> FamilySyntheticWeights(GgufMetadataView g, IReadOnlyDictionary<string, WeightInfo> weights) => [];

    /// <summary>Query and key/value head counts of layer <paramref name="layer"/> (used to split fused q/k/v tensors).</summary>
    protected virtual (int Heads, int KvHeads) HeadsOf(GgufMetadataView g, int layer)
    {
        int heads = g.Int("attention.head_count");
        return (heads, g.Int("attention.head_count_kv", heads));
    }

    /// <summary>The model-wide head size (key length, else hidden size over head count).</summary>
    protected static int HeadDimOf(GgufMetadataView g)
        => g.Has("attention.key_length") ? g.Int("attention.key_length") : g.Int("embedding_length") / GgufDecoderHyperParameters.LargestOf(g, "attention.head_count");

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(GgufMetadataView g)
    {
        var md = GgufDecoderHyperParameters.Common(g, Activation, RopeStyle, LayerNorm);
        md[GgufDecoderHyperParameters.Prefix + "layer_count"] = TrunkLayerCount(g);
        if (!Gated)
        {
            md[GgufDecoderHyperParameters.Prefix + "mlp.gated"] = false;
        }

        if (g.Int("expert_count", 0) > 0)
        {
            DecoderGgufMapping.AddMixtureOfExperts(md, GgufDecoderHyperParameters.Prefix, g, NormalizesExpertWeights);
        }

        AddHyperParameters(md, g);
        DecoderGgufMapping.AddRopeScaling(md, GgufDecoderHyperParameters.Prefix, g);
        return md;
    }

    /// <inheritdoc/>
    public string? MapWeightName(string name, GgufMetadataView g)
    {
        if (GlobalTensorOverrides is { } globalOverrides && globalOverrides.TryGetValue(name, out var overriddenGlobal))
        {
            return overriddenGlobal;
        }

        if (GgufTensorNames.MapGlobal(name) is { } globalName)
        {
            return globalName;
        }

        if (DecoderGgufMapping.RopeFactorTensors.TryGetValue(name, out var ropeName) || s_extraGlobal.TryGetValue(name, out ropeName))
        {
            return ropeName;
        }

        var m = GgufTensorNames.Block().Match(name);
        if (!m.Success)
        {
            return null;
        }

        string suffix = m.Groups[2].Value;
        int layer = int.Parse(m.Groups[1].Value);
        int trunkLayers = TrunkLayerCount(g);
        string layerPrefix = layer < trunkLayers ? $"layers.{layer}." : $"mtp.layers.{layer - trunkLayers}.";
        if (layer >= trunkLayers && s_nextn.TryGetValue(suffix, out var nextnName))
        {
            return layerPrefix + nextnName;
        }

        if (LayerTensorOverrides is { } overrides && overrides.TryGetValue(suffix, out var overridden))
        {
            return layerPrefix + overridden;
        }

        return DecoderGgufMapping.LayerTensors.TryGetValue(suffix, out var canonical) ? layerPrefix + canonical : null;
    }

    /// <summary>
    /// Decoder layers without the multi-token-prediction (NextN) blocks that some converters append after them (<c>nextn_predict_layers</c>).
    /// The NextN blocks are kept under <c>mtp.layers.{j}.</c> (the draft head, see <c>MtpDraftHead.FirstNextnModule</c>).
    /// </summary>
    protected static int TrunkLayerCount(GgufMetadataView g) => g.Int("block_count") - g.Int("nextn_predict_layers", 0);

    /// <summary>NextN-only tensors of an MTP block, GGUF suffix to canonical suffix (shared with the other decoder mappings).</summary>
    internal static IReadOnlyDictionary<string, string> NextnTensors => s_nextn;

    private static readonly Dictionary<string, string> s_nextn = new(StringComparer.Ordinal)
    {
        ["nextn.eh_proj.weight"] = "fc.weight",
        ["nextn.enorm.weight"] = "embedding_norm.weight",
        ["nextn.hnorm.weight"] = "hidden_norm.weight",
        ["nextn.shared_head_norm.weight"] = "output_norm.weight",
        ["nextn.shared_head_head.weight"] = "lm_head.weight",
        ["nextn.embed_tokens.weight"] = "token_embedding.weight",
    };

    /// <summary>Fused <c>attn_qkv</c> (q rows, then k, then v) and fused gate+up <c>ffn_up</c> (gate rows first) tensors are split.</summary>
    public IReadOnlyList<WeightInfo>? Split(string name, WeightInfo t, GgufMetadataView g)
    {
        var m = GgufTensorNames.Block().Match(name);
        if (!m.Success)
        {
            return null;
        }

        int layer = int.Parse(m.Groups[1].Value);
        int trunkLayers = TrunkLayerCount(g);
        string prefix = layer < trunkLayers ? $"layers.{layer}." : $"mtp.layers.{layer - trunkLayers}.";
        switch (m.Groups[2].Value)
        {
            case "attn_qkv.weight":
                return SplitQkv(t, prefix, "weight", g, layer);
            case "attn_qkv.bias":
                return SplitQkv(t, prefix, "bias", g, layer);
            case "ffn_up.weight":
                int ffnLength = g.Raw("feed_forward_length") is Array ? 0 : g.Int("feed_forward_length", 0);
                if (ffnLength == 0 || t.Shape[0].Fixed != 2 * ffnLength)
                {
                    return null;
                }

                return [t.RowSlice(prefix + "mlp.gate.weight", 0, ffnLength), t.RowSlice(prefix + "mlp.up.weight", ffnLength, ffnLength)];
            case "ffn_up_shexp.weight":
                // A shared expert stored with its gate rows first, then the up rows (Granite MoE shared).
                int sharedLength = g.Int("expert_shared_feed_forward_length", 0);
                if (sharedLength == 0 || t.Shape[0].Fixed != 2 * sharedLength)
                {
                    return null;
                }

                return [t.RowSlice(prefix + "mlp.shared.gate.weight", 0, sharedLength), t.RowSlice(prefix + "mlp.shared.up.weight", sharedLength, sharedLength)];
            default:
                return null;
        }
    }

    private WeightInfo[] SplitQkv(WeightInfo t, string prefix, string kind, GgufMetadataView g, int layer)
    {
        var (heads, kvHeads) = HeadsOf(g, layer);
        int headDim = HeadDimOf(g);
        int valueDim = g.Int("attention.value_length", headDim);
        return
        [
            t.RowSlice($"{prefix}attention.q.{kind}", 0, heads * headDim),
            t.RowSlice($"{prefix}attention.k.{kind}", heads * headDim, kvHeads * headDim),
            t.RowSlice($"{prefix}attention.v.{kind}", (heads + kvHeads) * headDim, kvHeads * valueDim),
        ];
    }

    /// <inheritdoc/>
    public WeightInfo MapWeight(WeightInfo w, GgufMetadataView g)
    {
        if (w.Shape.Rank == 1 && w.Name.EndsWith("mlp.shared.input_gate.weight", StringComparison.Ordinal))
        {
            return w.As(w.Name, new Shape(1, w.Shape[0]));
        }

        if (!PermutesQk)
        {
            return w;
        }

        int headCount = g.Int("attention.head_count");
        if (w.Name.EndsWith("attention.q.weight", StringComparison.Ordinal) || w.Name.EndsWith("attention.q.bias", StringComparison.Ordinal))
        {
            return DecoderGgufMapping.UnpermuteRows(w, headCount);
        }

        if (w.Name.EndsWith("attention.k.weight", StringComparison.Ordinal) || w.Name.EndsWith("attention.k.bias", StringComparison.Ordinal))
        {
            return DecoderGgufMapping.UnpermuteRows(w, g.Int("attention.head_count_kv", headCount));
        }

        return w;
    }

    /// <inheritdoc/>
    public IEnumerable<WeightInfo> SyntheticWeights(GgufMetadataView g, IReadOnlyDictionary<string, WeightInfo> weights)
        => DecoderGgufMapping.RopeFrequencyFactors(g, weights).Concat(FamilySyntheticWeights(g, weights));
}
