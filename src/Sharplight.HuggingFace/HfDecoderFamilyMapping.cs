using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>
/// Base of the small per-family mappings of Hugging Face decoder-only models that the generic canonical <c>decoder</c> graph expresses.
/// It supplies the shared hyper-parameters and a table-driven weight walk with Llama naming as the default; a family overrides the
/// names, the norm convention, the rotary pairing and any fused or transformed tensors it has.
/// </summary>
internal abstract partial class HfDecoderFamilyMapping : IHfArchitectureMapping
{
    /// <summary>Llama naming: per-layer tensor suffix to canonical suffix. Norm tensors (canonical <c>*_norm.weight</c>) go through <see cref="NormWeight"/>.</summary>
    protected static readonly IReadOnlyDictionary<string, string> LlamaLayerTensors = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["self_attn.q_proj.weight"] = "attention.q.weight",
        ["self_attn.k_proj.weight"] = "attention.k.weight",
        ["self_attn.v_proj.weight"] = "attention.v.weight",
        ["self_attn.o_proj.weight"] = "attention.o.weight",
        ["self_attn.q_proj.bias"] = "attention.q.bias",
        ["self_attn.k_proj.bias"] = "attention.k.bias",
        ["self_attn.v_proj.bias"] = "attention.v.bias",
        ["self_attn.o_proj.bias"] = "attention.o.bias",
        ["self_attn.q_norm.weight"] = "attention.q_norm.weight",
        ["self_attn.k_norm.weight"] = "attention.k_norm.weight",
        ["mlp.gate_proj.weight"] = "mlp.gate.weight",
        ["mlp.up_proj.weight"] = "mlp.up.weight",
        ["mlp.down_proj.weight"] = "mlp.down.weight",
        ["mlp.gate_proj.bias"] = "mlp.gate.bias",
        ["mlp.up_proj.bias"] = "mlp.up.bias",
        ["mlp.down_proj.bias"] = "mlp.down.bias",
        ["input_layernorm.weight"] = "input_norm.weight",
        ["input_layernorm.bias"] = "input_norm.bias",
        ["post_attention_layernorm.weight"] = "post_attention_norm.weight",
        ["post_attention_layernorm.bias"] = "post_attention_norm.bias",
    };

    /// <summary>Llama naming outside the layers.</summary>
    protected static readonly IReadOnlyDictionary<string, string> LlamaGlobalTensors = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["model.embed_tokens.weight"] = "token_embedding.weight",
        ["model.norm.weight"] = "output_norm.weight",
        ["model.norm.bias"] = "output_norm.bias",
        ["lm_head.weight"] = "lm_head.weight",
        ["lm_head.bias"] = "lm_head.bias",
    };

    [GeneratedRegex(@"^model\.layers\.(\d+)\.(.+)$")]
    private static partial Regex LlamaLayerName();

    /// <inheritdoc/>
    public string Architecture => "decoder";

    /// <inheritdoc/>
    public abstract IReadOnlyList<string> ModelTypes { get; }

    /// <summary>Whether the model uses mean-centred layer norm instead of RMS norm.</summary>
    protected virtual bool LayerNorm => false;

    /// <summary>Whether the MLP is gated (<c>act(gate) * up</c>); false for plain <c>act(up)</c> blocks.</summary>
    protected virtual bool Gated => true;

    /// <summary>Rotary pairing of the checkpoint's q/k rows: "neox" (rotate halves) or "gptj" (interleaved pairs).</summary>
    protected virtual string RopeStyle => "neox";

    /// <summary>Canonical activation name.</summary>
    protected virtual string Activation(HfConfig c) => DecoderHfMapping.Activation(c);

    /// <summary>Hidden size.</summary>
    protected virtual int HiddenSize(HfConfig c) => c.Int("hidden_size");

    /// <summary>Number of query heads.</summary>
    protected virtual int HeadCount(HfConfig c) => c.Int("num_attention_heads");

    /// <summary>Head size (explicit <c>head_dim</c>, else hidden size over head count).</summary>
    protected int HeadDimOf(HfConfig c) => c.GetInt("head_dim") ?? HiddenSize(c) / HeadCount(c);

    /// <summary>Training context length.</summary>
    protected virtual int? ContextLength(HfConfig c) => c.GetInt("max_position_embeddings");

    /// <summary>Number of decoder layers.</summary>
    protected virtual int LayerCount(HfConfig c) => c.Int("num_hidden_layers");

    /// <summary>Feed-forward width of the (dense) MLP.</summary>
    protected virtual int FeedForwardSize(HfConfig c) => c.Int("intermediate_size");

    /// <summary>Number of key/value heads.</summary>
    protected virtual int KvHeadCount(HfConfig c, int headCount) => c.GetInt("num_key_value_heads") ?? headCount;

    /// <summary>Epsilon of the norms.</summary>
    protected virtual float NormEpsilon(HfConfig c)
        => (float)(c.GetDouble("rms_norm_eps", "layer_norm_eps", "layer_norm_epsilon", "norm_epsilon") ?? 1e-5);

    /// <summary>Number of rotated dimensions of a head.</summary>
    protected virtual int RopeDimension(HfConfig c, int headDim) => DecoderHfMapping.RopeDim(c, headDim);

    /// <summary>RoPE base frequency.</summary>
    protected virtual float RopeBase(HfConfig c) => (float)(c.RopeDouble("rope_theta", "rope_theta", "rotary_emb_base") ?? 10000.0);

    /// <summary>Rope scaling described by the config (llama3 / yarn / linear / longrope): the attention magnitude factor. Families with their own scaling override both rope-scaling hooks.</summary>
    protected virtual void AddRopeScaling(Dictionary<string, object> md, HfConfig c) => DecoderHfMapping.AddRopeAttentionScale(md, "decoder.", c);

    /// <summary>The per-frequency factor weights of the rope scaling.</summary>
    protected virtual void AddRopeFactors(HfConfig c, HfWeightCollector collector) => DecoderHfMapping.AddRopeFactors(c, extended: false, collector);

    /// <summary>Family-specific hyper-parameters on top of the shared ones.</summary>
    protected virtual void AddHyperParameters(Dictionary<string, object> md, HfConfig c)
    {
    }

    /// <summary>Per-layer tensor names of this family (HF suffix to canonical suffix).</summary>
    protected virtual IReadOnlyDictionary<string, string> LayerTensors => LlamaLayerTensors;

    /// <summary>Tensor names outside the layers (HF name to canonical name).</summary>
    protected virtual IReadOnlyDictionary<string, string> GlobalTensors => LlamaGlobalTensors;

    /// <summary>The regex that splits a layer tensor name into layer index (group 1) and suffix (group 2).</summary>
    protected virtual Regex LayerName => LlamaLayerName();

    /// <summary>The canonical norm weight for a checkpoint norm tensor (default: unchanged; models with a <c>1 + w</c> convention fold it).</summary>
    protected virtual WeightInfo NormWeight(WeightInfo w, string name) => HfTransforms.Plain(w, name);

    /// <summary>Tensors that are neither mapped nor an error (buffers, unused heads).</summary>
    protected virtual bool IsIgnored(string name) => name.EndsWith("rotary_emb.inv_freq", StringComparison.Ordinal);

    /// <summary>Maps a layer tensor that needs more than a rename (fused projections, experts); false when it is not one.</summary>
    protected virtual bool TryMapSpecialLayerTensor(HfConfig c, string layerPrefix, string suffix, WeightInfo w, HfWeightCollector collector) => false;

    /// <summary>Called after every tensor was seen (e.g. to stack per-expert tensors).</summary>
    protected virtual void FinishWeights(HfConfig c, HfWeightCollector collector)
    {
    }

    /// <summary>An integer-array config value (e.g. <c>no_rope_layers</c>), or null.</summary>
    protected static int[]? IntArray(HfConfig c, string key)
        => c.Get(key) is { ValueKind: System.Text.Json.JsonValueKind.Array } array ? array.EnumerateArray().Select(e => (int)e.GetDouble()).ToArray() : null;

    /// <summary>
    /// Sliding-window attention described by <c>layer_types</c> (<c>sliding_attention</c> entries): adds the window and the per-layer
    /// flags. Returns the flags, or null when the config has no window or no sliding layers.
    /// </summary>
    protected static int[]? AddSlidingLayers(Dictionary<string, object> md, HfConfig c)
    {
        if (c.GetInt("sliding_window") is not > 0 || c.GetStringArray("layer_types") is not { } layerTypes)
        {
            return null;
        }

        var flags = HfMappingHelpers.SlidingFlags(layerTypes);
        if (!flags.Any(flag => flag != 0))
        {
            return null;
        }

        md["decoder.attention.sliding_window"] = c.Int("sliding_window");
        md["decoder.attention.sliding_layers"] = flags;
        return flags;
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HfImportOptions options)
    {
        const string prefix = "decoder.";
        int hidden = HiddenSize(c);
        int headCount = HeadCount(c);
        int headDim = HeadDimOf(c);
        var md = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [prefix + "layer_count"] = LayerCount(c),
            [prefix + "hidden_size"] = hidden,
            [prefix + "ffn_hidden_size"] = FeedForwardSize(c),
            [prefix + "norm_eps"] = NormEpsilon(c),
            [prefix + "attention.head_count"] = headCount,
            [prefix + "attention.kv_head_count"] = KvHeadCount(c, headCount),
            [prefix + "attention.head_dim"] = headDim,
            [prefix + "rope.dim"] = RopeDimension(c, headDim),
            [prefix + "rope.base"] = RopeBase(c),
            [prefix + "rope.style"] = RopeStyle,
            [prefix + "mlp.activation"] = Activation(c),
        };
        if (LayerNorm)
        {
            md[prefix + "norm.type"] = "layer";
        }

        if (!Gated)
        {
            md[prefix + "mlp.gated"] = false;
        }

        if (ContextLength(c) is { } contextLength)
        {
            md[prefix + "context_length"] = contextLength;
        }

        AddRopeScaling(md, c);
        AddHyperParameters(md, c);
        return md;
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HfImportOptions options)
    {
        var collector = new HfWeightCollector();
        bool tied = c.GetBool("tie_word_embeddings") ?? false;
        foreach (var (name, tensor) in tensors.OrderBy(p => p.Key, StringComparer.Ordinal))
        {
            if (IsIgnored(name))
            {
                continue;
            }

            var w = tensor.ToWeightInfo();
            if (GlobalTensors.TryGetValue(name, out var globalName))
            {
                if (!(tied && name == "lm_head.weight"))
                {
                    collector.Add(IsNorm(globalName) ? NormWeight(w, globalName) : HfTransforms.Plain(w, globalName));
                }

                continue;
            }

            var match = LayerName.Match(name);
            // A multi-token-prediction layer's tensor this family does not know is dropped (the head is then not built, or drafts
            // worse; verification keeps the output exact) instead of failing the import, which never needed those layers.
            if (!match.Success || (!TryMapLayerTensor(c, match.Groups[1].Value, match.Groups[2].Value, w, collector) && !IsMtpLayer(c, match.Groups[1].Value)))
            {
                collector.AddUnmapped(name);
            }
        }

        FinishWeights(c, collector);
        collector.ThrowIfUnmapped(c.ModelType);
        AddRopeFactors(c, collector);
        return collector.Weights;
    }

    private bool IsMtpLayer(HfConfig c, string layer)
        => c.GetInt("num_hidden_layers") is not null && int.TryParse(layer, out int index) && index >= LayerCount(c);

    private bool TryMapLayerTensor(HfConfig c, string layer, string suffix, WeightInfo w, HfWeightCollector collector)
    {
        // Multi-token-prediction (NextN) layers after the decoder's: an ordinary layer plus the NextN tensors, below mtp.layers.{j}.
        string prefix = $"layers.{layer}.";
        if (IsMtpLayer(c, layer))
        {
            prefix = $"mtp.layers.{int.Parse(layer) - LayerCount(c)}.";
            if (HfNextnTensors.Suffixes.TryGetValue(suffix, out var nextn))
            {
                if (nextn is not null)
                {
                    collector.Add(IsNorm(nextn) ? NormWeight(w, prefix + nextn) : HfTransforms.Plain(w, prefix + nextn));
                }

                return true;
            }
        }

        if (TryMapSpecialLayerTensor(c, prefix, suffix, w, collector))
        {
            return true;
        }

        if (!LayerTensors.TryGetValue(suffix, out var canonical))
        {
            return false;
        }

        collector.Add(IsNorm(canonical) ? NormWeight(w, prefix + canonical) : HfTransforms.Plain(w, prefix + canonical));
        return true;
    }

    private static bool IsNorm(string canonical) => canonical.EndsWith("_norm.weight", StringComparison.Ordinal);

    /// <summary>A copy of <paramref name="baseTable"/> with <paramref name="changes"/> applied; a null value removes the entry.</summary>
    protected static IReadOnlyDictionary<string, string> With(IReadOnlyDictionary<string, string> baseTable, params (string Key, string? Value)[] changes)
    {
        var table = new Dictionary<string, string>(baseTable, StringComparer.Ordinal);
        foreach (var (key, value) in changes)
        {
            if (value is null)
            {
                table.Remove(key);
            }
            else
            {
                table[key] = value;
            }
        }

        return table;
    }
}
