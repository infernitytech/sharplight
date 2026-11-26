using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// The attention mixer of the <see cref="BlockResidualDecoderAdapter"/> (Inkling 2): rope-less grouped-query attention over a KV cache whose
/// queries, keys and values may come from three different inputs, with per-token attention sinks, an output gate, L2-clipped queries and keys,
/// RMS-normed values, short convolutions of the keys and values and a query-conditioned relative position bias.
/// </summary>
/// <remarks>
/// Weights below the layer prefix:
/// <list type="bullet">
/// <item><c>attention.{q,k,v,o}.weight</c>; <c>attention.gate.weight</c> [heads·D, H]: the attention output is multiplied by the sigmoid of this
/// projection of the query input before <c>o</c>; <c>attention.sinks.weight</c> [heads, H]: one sink logit per token and head (CachedAttention
/// <c>sinks</c> [T, heads]).</item>
/// <item>Queries: <c>q = clip(q ⊙ attention.q_clip.scale)</c> per head ([heads, D]), where clip(x) = x · min(1, c / (‖x‖ + 10⁻⁶)) with
/// c = <c>attention.qk_clip</c>; keys: clip(k) without a scale; values: a weight-less RMS norm per head (epsilon <c>norm_eps</c>).</item>
/// <item>Key and value convolutions before the head split, <c>y = (x + conv(x)) / √(K + 1)</c>: static ones with <c>attention.{k,v}_conv.weight</c>
/// [C, K]; with <c>attention.{k,v}_conv.taps.weight</c> [G·K, H] the taps are the static kernel plus a per-token projection of the key (value)
/// input shared by the C / G channels of each of G groups, and <c>y = (x + conv(x)) / √(2K + 1)</c> (the window of the last K − 1 rows comes from
/// a causal convolution with one-hot kernels, so the multi-sequence and rollback forms of <c>CausalConv1d</c> carry it).</item>
/// <item>Relative bias: <c>r = clip(attention.relative.weight · x_q ⊙ attention.relative_clip.scale)</c> per head ([heads, R], threshold
/// <c>attention.relative_clip</c>), mixed by <c>attention.relative_bias.weight</c> [E, R] into one bias per backward distance 0 ≤ d &lt; E
/// (CachedAttention <c>relative_bias</c>).</item>
/// <item>On the layers without a sliding window, the log-floor scaling (<c>attention.log_scaling_floor</c> &gt; 0, see
/// <see cref="AttentionTemperature.LogFloor"/>) multiplies the clipped queries and relative states.</item>
/// </list>
/// The scores are scaled by 1/√D. Sliding layers attend to their last <c>attention.sliding_window</c> keys.
/// </remarks>
internal sealed class GatedSinkAttention
{
    private const float ClipEps = 1e-6f;
    private readonly int _heads;
    private readonly int _kvHeads;
    private readonly int _headDim;
    private readonly float _eps;
    private readonly int _window;
    private readonly int[] _slidingLayers;
    private readonly int[] _layerHeads;
    private readonly int[] _layerKvHeads;
    private readonly float _qkClip;
    private readonly float _relativeClip;
    private readonly float _logScalingFloor;
    private readonly float _logScalingAlpha;

    /// <summary>Reads the <c>attention.*</c> hyper-parameters of a source.</summary>
    public GatedSinkAttention(ModelSource src)
    {
        _heads = src.Arch("attention.head_count", 0);
        _kvHeads = src.Arch("attention.kv_head_count", _heads);
        _headDim = src.Arch("attention.head_dim", 0);
        _eps = src.Arch("norm_eps", 1e-6f);
        _window = src.Arch("attention.sliding_window", 0);
        _slidingLayers = src.Arch("attention.sliding_layers", Array.Empty<int>());
        _layerHeads = src.Arch("attention.layer_head_counts", Array.Empty<int>());
        _layerKvHeads = src.Arch("attention.layer_kv_head_counts", Array.Empty<int>());
        _qkClip = src.Arch("attention.qk_clip", 0f);
        _relativeClip = src.Arch("attention.relative_clip", 0f);
        _logScalingFloor = src.Arch("attention.log_scaling_floor", 0f);
        _logScalingAlpha = src.Arch("attention.log_scaling_alpha", 0f);
    }

    /// <summary>Whether layer <paramref name="il"/> has a sliding window.</summary>
    public bool Sliding(int il) => _window > 0 && il < _slidingLayers.Length && _slidingLayers[il] != 0;

    private int HeadsOf(int il) => il < _layerHeads.Length && _layerHeads[il] > 0 ? _layerHeads[il] : _heads;

    private int KvHeadsOf(int il) => il < _layerKvHeads.Length && _layerKvHeads[il] > 0 ? _layerKvHeads[il] : _kvHeads;

    /// <summary>The states of one pass of layer <paramref name="il"/>: key / value caches and the key / value convolution windows.</summary>
    /// <param name="src">The model source (for the convolution weights).</param>
    /// <param name="group">The placement group (the layer).</param>
    /// <param name="pass">The pass: 0 names the states after the layer (<c>{group}.k_cache</c>), later passes <c>{group}.pass{n}.k_cache</c>.</param>
    /// <param name="il">The layer index.</param>
    /// <param name="cacheType">The KV cache element type.</param>
    public IReadOnlyList<StateDefinition> DefineStates(ModelSource src, string group, int pass, int il, DataType cacheType)
    {
        string prefix = PassPrefix(group, pass);
        var shape = new Shape(Dim.Sym(WellKnown.ContextSymbol), KvHeadsOf(il), _headDim);
        var type = AdapterOptions.KvRowType(cacheType, _headDim);
        string p = $"layers.{il}.";
        return
        [
            new StateDefinition { Name = $"{prefix}.k_cache", Type = type, Shape = shape, Kind = StateKind.KeyCache, Group = group },
            new StateDefinition { Name = $"{prefix}.v_cache", Type = type, Shape = shape, Kind = StateKind.ValueCache, Group = group },
            ConvState(src, group, pass, p + "attention.k_conv"),
            ConvState(src, group, pass, p + "attention.v_conv"),
        ];
    }

    /// <summary>The state name prefix of pass <paramref name="pass"/> of a layer: the group itself for the first pass.</summary>
    public static string PassPrefix(string group, int pass) => pass == 0 ? group : $"{group}.pass{pass}";

    // The window of a key / value convolution: K − 1 rows of C channels (static taps) or of (K − 1)·C (the lag rows of per-token taps).
    private static StateDefinition ConvState(ModelSource src, string group, int pass, string conv)
    {
        var w = src.Weight(conv + ".weight") ?? throw new InvalidDataException($"block_residual_decoder: missing {conv}.weight.");
        int channels = (int)w.Shape[0].Fixed;
        int taps = (int)w.Shape[1].Fixed;
        bool perToken = src.Weight(conv + ".taps.weight") is not null;
        string name = conv[(conv.LastIndexOf('.') + 1)..] + "_state";
        return RecurrentStates.ConvWindow(group, pass == 0 ? name : $"pass{pass}.{name}", taps - 1, perToken ? (taps - 1) * channels : channels);
    }

    /// <summary>Emits the mixer of layer <paramref name="il"/>.</summary>
    /// <param name="b">The graph builder.</param>
    /// <param name="hq">The normalized query input [T, H] (also the input of the relative bias, the sinks and the output gate).</param>
    /// <param name="hk">The normalized key input [T, H].</param>
    /// <param name="hv">The normalized value input [T, H].</param>
    /// <param name="positions">Positions [T, 1] (i32).</param>
    /// <param name="il">The layer index.</param>
    /// <param name="states">The pass's states, as <see cref="DefineStates"/> returns them.</param>
    /// <param name="constants">The graph's constants (the one-hot kernels of the per-token convolutions).</param>
    public Value Build(GraphBuilder b, Value hq, Value hk, Value hv, Value positions, int il, IReadOnlyList<StateDefinition> states, ConstantCache constants)
    {
        string p = $"layers.{il}.";
        int heads = HeadsOf(il);
        int kvHeads = KvHeadsOf(il);
        bool sliding = Sliding(il);
        Value? temperature = !sliding && _logScalingFloor > 0 ? AttentionTemperature.LogFloor(b, positions, _logScalingAlpha, _logScalingFloor) : null;

        var q = Clip(b, b.Mul(b.Reshape(b.Linear(hq, p + "attention.q.weight"), 0, heads, _headDim), b.Weight(p + "attention.q_clip.scale")), _qkClip);
        if (temperature is not null)
        {
            q = b.Mul(q, temperature);
        }

        var k = Convolved(b, b.Linear(hk, p + "attention.k.weight"), hk, p + "attention.k_conv", states[2], constants);
        var v = Convolved(b, b.Linear(hv, p + "attention.v.weight"), hv, p + "attention.v_conv", states[3], constants);
        k = Clip(b, b.Reshape(k, 0, kvHeads, _headDim), _qkClip);
        v = b.RmsNorm(b.Reshape(v, 0, kvHeads, _headDim), null, _eps);

        var relativeBias = b.Weight(p + "attention.relative_bias.weight");
        long r = relativeBias.Shape[1].Fixed;
        var relative = Clip(b, b.Mul(b.Reshape(b.Linear(hq, p + "attention.relative.weight"), 0, heads, r), b.Weight(p + "attention.relative_clip.scale")),
            _relativeClip);
        var bias = b.MatMul(relative, relativeBias, $"{p}relative_bias");                                                            // [T, heads, E]
        if (temperature is not null)
        {
            bias = b.Mul(bias, temperature);
        }

        var sinks = b.Linear(hq, p + "attention.sinks.weight", name: $"{p}attn_sinks");                                              // [T, heads]

        b.CacheWrite(b.State(states[0]), k);
        b.CacheWrite(b.State(states[1]), v);
        var attrs = new List<(string, object)>
        {
            ("scale", 1f / MathF.Sqrt(_headDim)), ("offset", WellKnown.PastScalar), ("sinks", true), ("relative_bias", (int)relativeBias.Shape[0].Fixed),
        };
        if (sliding)
        {
            attrs.Add(("window", _window));
        }

        b.RuntimeScalar(WellKnown.PastScalar);
        var attn = b.Emit1("CachedAttention", [q, b.State(states[0]), b.State(states[1]), sinks, bias], Attributes.From([.. attrs]), $"{p}attn");
        var gate = b.Sigmoid(b.Linear(hq, p + "attention.gate.weight"));
        return b.Linear(b.Mul(b.Reshape(attn, 0, heads * _headDim), gate), p + "attention.o.weight", name: $"{p}attn_out");
    }

    /// <summary><c>(x + conv(x)) / √(K + 1)</c> with the static depthwise kernel <paramref name="weightName"/> [C, K] and its window state.</summary>
    public static Value ResidualConv(GraphBuilder b, Value x, string weightName, StateDefinition window)
    {
        var w = b.Weight(weightName);
        var state = b.State(window);
        var conv = b.Emit("CausalConv1d", [x, w, state], Attributes.From(("activation", "none")), writes: [state])[0];
        return b.Scale(b.Add(x, conv), 1f / MathF.Sqrt(w.Shape[1].Fixed + 1));
    }

    // A key / value convolution: static, or with per-token taps computed from the projection's input.
    private static Value Convolved(GraphBuilder b, Value x, Value input, string conv, StateDefinition window, ConstantCache constants)
    {
        if (!b.HasWeight(conv + ".taps.weight"))
        {
            return ResidualConv(b, x, conv + ".weight", window);
        }

        var kernel = b.Weight(conv + ".weight");                                                                                     // [C, K]
        int channels = (int)kernel.Shape[0].Fixed;
        int taps = (int)kernel.Shape[1].Fixed;
        var tapWeight = b.Weight(conv + ".taps.weight");                                                                             // [G·K, H]
        int groups = (int)(tapWeight.Shape[0].Fixed / taps);
        int width = channels / groups;

        // The rows x[t − K + 1 + j] for j < K − 1 (convolution order) from one depthwise convolution of K − 1 copies of x with one-hot kernels.
        var copies = b.Concat(1, [.. Enumerable.Repeat(x, taps - 1)]);                                                              // [T, (K−1)·C]
        var oneHot = constants.F32($"one_hot_taps_{taps}_{channels}", new Shape((taps - 1) * channels, taps), () =>
        {
            var values = new float[(taps - 1) * channels * taps];
            for (int j = 0; j < taps - 1; j++)
            {
                for (int c = 0; c < channels; c++)
                {
                    values[((j * channels) + c) * taps + j] = 1f;
                }
            }

            return values;
        });
        var state = b.State(window);
        var lagged = b.Emit("CausalConv1d", [copies, oneHot, state], Attributes.From(("activation", "none")), writes: [state])[0];
        var rows = b.Concat(1, b.Reshape(lagged, 0, taps - 1, channels), b.Reshape(x, 0, 1, channels));                           // [T, K, C]

        // Taps per row, channel and tap: the static kernel [K, C] plus the row's projection of its group.
        var dynamic = b.Reshape(b.Permute(b.Reshape(b.Linear(input, conv + ".taps.weight"), 0, groups, taps), 0, 2, 1), 0, taps, groups, 1);  // [T, K, G, 1]
        var weights = b.Add(b.Reshape(b.Permute(kernel, 1, 0), taps, groups, width), dynamic);                                    // [T, K, G, C/G]
        var products = b.Mul(b.Reshape(rows, 0, taps, groups, width), weights);
        var summed = b.Reshape(b.Emit1("ReduceSum", [b.Permute(products, 0, 2, 3, 1)]), 0, channels);                             // [T, C]
        return b.Scale(b.Add(x, summed), 1f / MathF.Sqrt(2 * taps + 1));
    }

    // x · min(1, c / (‖x‖ + 10⁻⁶)) over the last axis.
    private static Value Clip(GraphBuilder b, Value x, float threshold)
    {
        var norm = b.Unary("Sqrt", b.Emit1("ReduceSum", [b.Unary("Square", x)]));
        var factor = b.Emit1("Clip", [b.Scale(b.Unary("Recip", b.Scale(norm, 1f, ClipEps)), threshold)], Attributes.From(("min", float.NegativeInfinity), ("max", 1f)));
        return b.Mul(x, factor);
    }
}
