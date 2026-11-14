using System.Buffers.Binary;
using System.Runtime.CompilerServices;
using System.Text.Json;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>limite</c> (Limite, <c>LimiteForCausalLM</c> remote code) onto the canonical decoder, with the folds of llama.cpp's converter:
/// the scalar projection scales (<c>qkv_scale</c>, <c>o_scale</c>, <c>post_lambda_attn</c> with <c>attn_gate_scale</c>, <c>post_lambda_mlp</c>) are
/// multiplied into q/k/v, o and down; the gates that read the first channels of the layer input (<c>attn_gate</c>, <c>ve_gate</c>) are padded to the
/// hidden width; the value embeddings keep the first key/value heads, padded to the head width and scaled by <c>ve_gate_scale</c>; <c>xsa_alpha</c>
/// goes through its tanh; the MUDD dense connections (<c>model.mudd.*</c>) become per-layer <c>dense.*</c> weights. The remote code rotates each
/// pair by −θ with frequencies base^(−i/(n−1)): q and k swap the lanes of their rotated pairs (the scores do not change) and the rotation runs at
/// +θ with base^(n/(n−1)). Its bfloat16 rotary tables are not reproduced (the rotation is f32). Weight-less RMS norms with the f32 machine epsilon.
/// </summary>
internal sealed class LimiteHfMapping : HfDecoderFamilyMapping
{
    private const string Staging = "limite.staging.";
    private const float Float32Epsilon = 1.1920929e-7f;

    private static readonly ConditionalWeakTable<HfWeightCollector, Dictionary<string, WeightInfo>> s_pending = [];

    private static readonly HashSet<string> s_pendingSuffixes = new(StringComparer.Ordinal)
    {
        "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight", "mlp.down_proj.weight",
        "self_attn.qkv_scale", "self_attn.o_scale", "post_lambda_attn", "post_lambda_mlp", "resid_lambda_attn", "resid_lambda_mlp",
        "self_attn.attn_gate", "self_attn.ve_gate", "self_attn.xsa_alpha",
    };

    private static readonly IReadOnlyDictionary<string, string> s_globalTensors = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["model.embed_tokens.weight"] = "token_embedding.weight",
        ["lm_head.weight"] = "lm_head.weight",
        ["model.value_embeds.weight"] = Staging + "value_embeds",
        ["model.mudd.dense1"] = Staging + "dense1",
        ["model.mudd.dense2"] = Staging + "dense2",
        ["model.mudd.dense2_mlp"] = Staging + "dense2_mlp",
        ["model.mudd.bias"] = Staging + "bias",
        ["model.mudd.bias_mlp"] = Staging + "bias_mlp",
    };

    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["limite"];

    /// <inheritdoc/>
    protected override string RopeStyle => "gptj";

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> GlobalTensors => s_globalTensors;

    /// <inheritdoc/>
    protected override float NormEpsilon(HfConfig c) => Float32Epsilon;

    /// <inheritdoc/>
    protected override int RopeDimension(HfConfig c, int headDim) => 2 * c.Int("rope_n_pairs");

    /// <inheritdoc/>
    protected override float RopeBase(HfConfig c)
    {
        int pairs = c.Int("rope_n_pairs");
        return (float)Math.Pow(c.Double("rope_base_local"), pairs / (pairs - 1.0));
    }

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, HfConfig c)
    {
        Validate(c);
        const string prefix = "decoder.";
        int layers = LayerCount(c);
        var global = IntArray(c, "global_layers") ?? [];
        int[] sliding = [.. Enumerable.Range(0, layers).Select(il => global.Contains(il) ? 0 : 1)];
        md[prefix + "attention.sliding_window"] = c.Int("sliding_window") + 1;   // keys k >= q - window: window + 1 tokens
        md[prefix + "attention.sliding_layers"] = sliding;
        md[prefix + "rope.layers"] = sliding;                                     // the global layers are NoPE
        md[prefix + "attention.scale"] = (float)c.Double("attention_softmax_scale");
        md[prefix + "attention.qk_norm_unweighted"] = true;
        md[prefix + "attention.exclusive_eps"] = c.GetBool("xsa") == true ? (float)(c.GetDouble("xsa_normalize_eps") ?? 0.0) : 0f;
        md[prefix + "embedding_norm"] = true;
        if (c.Get("softcap_logits") is { ValueKind: JsonValueKind.Object } cap)
        {
            md[prefix + "logits.sigmoid_cap"] = new[] { "a", "b", "c" }.Select(k => (float)cap.GetProperty(k).GetDouble()).ToArray();
        }

        if (Taps(c) is { Count: > 0 } taps)
        {
            int count = c.Int("mudd_taps");
            var flat = Enumerable.Repeat(-1, layers * count).ToArray();
            foreach (var (layer, layerTaps) in taps)
            {
                layerTaps.CopyTo(flat, layer * count);
            }

            md[prefix + "dense_connections.tap_count"] = count;
            md[prefix + "dense_connections.taps"] = flat;
        }
    }

    /// <inheritdoc/>
    protected override bool TryMapSpecialLayerTensor(HfConfig c, string layerPrefix, string suffix, WeightInfo w, HfWeightCollector collector)
    {
        if (!s_pendingSuffixes.Contains(suffix))
        {
            return false;
        }

        s_pending.GetValue(collector, _ => new Dictionary<string, WeightInfo>(StringComparer.Ordinal))[layerPrefix + suffix] = w;
        return true;
    }

    /// <inheritdoc/>
    protected override void FinishWeights(HfConfig c, HfWeightCollector collector)
    {
        var pending = s_pending.TryGetValue(collector, out var p) ? p : [];
        s_pending.Remove(collector);
        int hidden = HiddenSize(c);
        int heads = HeadCount(c);
        int kvHeads = KvHeadCount(c, heads);
        int headDim = HeadDimOf(c);
        int pairs = c.Int("rope_n_pairs");
        var xsaLayers = c.GetBool("xsa") == true ? IntArray(c, "xsa_layers") ?? [] : [];
        float gateScale = (float)(c.GetDouble("attn_gate_scale") ?? 1.0);
        for (int il = 0; il < LayerCount(c); il++)
        {
            string hf = $"layers.{il}.";
            WeightInfo Pending(string suffix) => pending.GetValueOrDefault(hf + suffix) ?? throw new InvalidDataException($"limite: missing model.layers.{il}.{suffix}.");
            float Scalar(string suffix) => HfTransforms.DecodeF32(Pending(suffix))[0];

            collector.Add(Folded(Pending("self_attn.q_proj.weight"), hf + "attention.q.weight", () => Scalar("self_attn.qkv_scale"), SwapPairs(headDim, pairs)));
            collector.Add(Folded(Pending("self_attn.k_proj.weight"), hf + "attention.k.weight", () => Scalar("self_attn.qkv_scale"), SwapPairs(headDim, pairs)));
            collector.Add(Folded(Pending("self_attn.v_proj.weight"), hf + "attention.v.weight", () => Scalar("self_attn.qkv_scale")));
            collector.Add(Folded(Pending("self_attn.o_proj.weight"), hf + "attention.o.weight",
                () => Scalar("self_attn.o_scale") * Scalar("post_lambda_attn") * gateScale));
            collector.Add(Folded(Pending("mlp.down_proj.weight"), hf + "mlp.down.weight", () => Scalar("post_lambda_mlp")));
            collector.Add(HfTransforms.ToF32(Pending("resid_lambda_attn"), hf + "attention.residual_alpha", new Shape(1)));
            collector.Add(HfTransforms.ToF32(Pending("resid_lambda_mlp"), hf + "mlp.residual_alpha", new Shape(1)));
            if (pending.GetValueOrDefault(hf + "self_attn.attn_gate") is { } gate)
            {
                collector.Add(PaddedColumns(gate, hf + "attention.gate.weight", (int)gate.Shape[0].Fixed, hidden));
            }

            if (pending.GetValueOrDefault(hf + "self_attn.ve_gate") is { } valueGate)
            {
                collector.Add(PaddedColumns(valueGate, hf + "attention.value_embedding_gate.weight", kvHeads, hidden));
            }

            if (xsaLayers.Contains(il) && pending.GetValueOrDefault(hf + "self_attn.xsa_alpha") is { } alpha)
            {
                collector.Add(HfTransforms.ToF32(alpha, hf + "attention.exclusive.weight", map: MathF.Tanh));
            }
        }

        if (collector.Take(Staging + "value_embeds") is { } valueEmbeds)
        {
            collector.Add(ValueEmbeddings(c, valueEmbeds, kvHeads, headDim));
        }

        var dense1 = collector.Take(Staging + "dense1");
        var dense2 = collector.Take(Staging + "dense2");
        var dense2Mlp = collector.Take(Staging + "dense2_mlp");
        var bias = collector.Take(Staging + "bias");
        var biasMlp = collector.Take(Staging + "bias_mlp");
        if (dense1 is null || dense2 is null || dense2Mlp is null || bias is null || biasMlp is null)
        {
            return;
        }

        int inner = (int)dense1.Shape[0].Fixed;
        foreach (var (layer, taps) in Taps(c) ?? [])
        {
            // Rows [0, n) mix the attention input, rows [n, 2n) the residual stream.
            int n = taps.Length;
            string prefix = $"layers.{layer}.dense.";
            collector.Add(HfTransforms.ToF32(dense1, prefix + "down.weight"));
            var up = WeightInfo.ConcatRows(prefix + "up.weight",
                [dense2.RowSlice("", layer, 1).As("", new Shape(n, inner)), dense2Mlp.RowSlice("", layer, 1).As("", new Shape(n, inner))]);
            var upBias = WeightInfo.ConcatRows(prefix + "up.bias", [bias.RowSlice("", layer, 1).As("", new Shape(n)), biasMlp.RowSlice("", layer, 1).As("", new Shape(n))]);
            collector.Add(HfTransforms.ToF32(up, prefix + "up.weight"));
            collector.Add(HfTransforms.ToF32(upBias, prefix + "up.bias"));
        }
    }

    // The layouts and conventions llama.cpp's converter accepts; anything else would load into a different model.
    private static void Validate(HfConfig c)
    {
        (string Key, string Value)[] expected =
        [
            ("pos_mode", "rope"), ("qk_norm", "rms_pre_rope"), ("rope_style", "interleaved_pairs_odd_lane_sign_flip"), ("mlp_type", "swiglu"),
            ("attn_gate_applied", "per_head_before_o_proj"), ("ve_head_slice", "first_num_key_value_heads"), ("mudd_r_site", "resid"),
            ("rms_norm_eps_mode", "torch_finfo_default"),
        ];
        foreach (var (key, value) in expected)
        {
            if (c.GetString(key) is { } actual && actual != value)
            {
                throw new NotSupportedException($"limite: unsupported {key} '{actual}' (expected '{value}').");
            }
        }

        if (c.GetBool("rms_norm_has_weight") == true || c.GetBool("rope_per_layer") == true || c.GetBool("global_nope") == false
            || (c.GetDouble("final_softcap") ?? 0.0) != 0.0 || (c.GetBool("mudd") == true && c.GetBool("mudd_mlp") != true))
        {
            throw new NotSupportedException("limite: unsupported norm, rope, soft-cap or MUDD configuration.");
        }
    }

    // The MUDD taps per layer (mudd_tap_idx: {"layer": [taps]}), or null without dense connections.
    private static List<(int Layer, int[] Taps)>? Taps(HfConfig c)
    {
        if (c.GetBool("mudd") != true || c.Get("mudd_tap_idx") is not { ValueKind: JsonValueKind.Object } taps)
        {
            return null;
        }

        return [.. taps.EnumerateObject().Select(p => (int.Parse(p.Name, System.Globalization.CultureInfo.InvariantCulture),
            p.Value.EnumerateArray().Select(e => (int)e.GetDouble()).ToArray())).OrderBy(t => t.Item1)];
    }

    // Row order of a q/k projection whose rotated pairs swap lanes: within each head the first 2·pairs rows go (1, 0, 3, 2, ...).
    private static Func<int, int> SwapPairs(int headDim, int pairs)
        => row => row % headDim < 2 * pairs ? row ^ 1 : row;

    // w · scale with rows taken through `rowOf` (destination row → source row); kept in BF16 when stored so, F32 otherwise.
    private static WeightInfo Folded(WeightInfo w, string name, Func<float> scale, Func<int, int>? rowOf = null)
    {
        int rows = (int)w.Shape[0].Fixed;
        int columns = (int)(w.Shape.Elements() / rows);
        var type = w.Type == DataType.BF16 ? DataType.BF16 : DataType.F32;
        return WeightInfo.Derived(name, type, w.Shape, () =>
        {
            var source = HfTransforms.DecodeF32(w);
            float factor = scale();
            var values = new float[source.Length];
            for (int r = 0; r < rows; r++)
            {
                int from = rowOf?.Invoke(r) ?? r;
                for (int j = 0; j < columns; j++)
                {
                    values[(long)r * columns + j] = source[(long)from * columns + j] * factor;
                }
            }

            return Encode(values, type);
        });
    }

    // The first `rows` rows of a gate [R, C] that reads the first C channels, zero-padded to [rows, width] (F32).
    private static WeightInfo PaddedColumns(WeightInfo w, string name, int rows, int width)
    {
        int columns = (int)w.Shape[1].Fixed;
        return WeightInfo.Derived(name, DataType.F32, new Shape(rows, width), () =>
        {
            var source = HfTransforms.DecodeF32(w);
            var values = new float[rows * width];
            for (int r = 0; r < rows; r++)
            {
                Array.Copy(source, r * columns, values, r * width, columns);
            }

            return HfTransforms.EncodeF32(values);
        });
    }

    // value_embeds [V, stored · ve_dim] → the first key/value heads, each zero-padded to the head width and scaled by ve_gate_scale: [V, KvH · D].
    private static WeightInfo ValueEmbeddings(HfConfig c, WeightInfo w, int kvHeads, int headDim)
    {
        int veDim = c.GetInt("ve_dim") ?? headDim;
        float scale = (float)(c.GetDouble("ve_gate_scale") ?? 1.0);
        long vocab = w.Shape[0].Fixed;
        int stored = (int)w.Shape[1].Fixed;
        var type = w.Type == DataType.BF16 ? DataType.BF16 : DataType.F32;
        return WeightInfo.Derived("value_embedding.weight", type, new Shape(vocab, kvHeads * headDim), () =>
        {
            var source = HfTransforms.DecodeF32(w);
            var values = new float[vocab * kvHeads * headDim];
            for (long t = 0; t < vocab; t++)
            {
                for (int h = 0; h < kvHeads; h++)
                {
                    for (int d = 0; d < veDim; d++)
                    {
                        values[(t * kvHeads + h) * headDim + d] = source[t * stored + h * veDim + d] * scale;
                    }
                }
            }

            return Encode(values, type);
        });
    }

    private static byte[] Encode(float[] values, DataType type)
    {
        if (type != DataType.BF16)
        {
            return HfTransforms.EncodeF32(values);
        }

        var bytes = new byte[values.LongLength * 2];
        for (long i = 0; i < values.LongLength; i++)
        {
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(checked((int)(i * 2))), HfFp8Codec.ToBFloat16(values[i]));
        }

        return bytes;
    }
}
