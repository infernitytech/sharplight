using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Grouped-query attention mixer of the <see cref="HybridAdapter"/> over a KV cache: weights <c>attention.{q,k,v,o}.weight</c>, optional per-head q/k RMS norms (<c>attention.{q,k}_norm.weight</c> [head_dim], or [heads, head_dim] / [kv_heads, head_dim] with a weight of its own for every head), optional rotary
/// embeddings (<c>attention.rope</c>, with the frequency factors <c>rope.freq_factors</c> when the model has them), scale <c>attention.scale</c> (default 1/sqrt(head_dim)),
/// sliding-window layers (<c>attention.sliding_window</c> and the per-layer flags <c>attention.sliding_layers</c>; their rotation uses <c>rope.local_base</c>
/// and no frequency factors) and an optional output gate <c>attention.gate.weight</c> (sigmoid of a projection of the layer input, one scalar per head
/// when it is as wide as the head count, else one per channel; LFM3).
/// </summary>
/// <remarks>
/// Further options (Inkling), each enabled by its weights or hyper-parameters:
/// <list type="bullet">
/// <item>Per-layer head counts <c>attention.layer_head_counts</c> / <c>attention.layer_kv_head_counts</c> (int[], 0 = the global count), and
/// <c>attention.scale_by_head_dim</c> (bool: scale 1/head_dim instead of 1/sqrt(head_dim), for q/k-normed heads).</item>
/// <item>Residual short convolutions of the key and value projections, before the head split: <c>attention.k_conv.weight</c> /
/// <c>attention.v_conv.weight</c> [C, K] f32, <c>k = k + causal_conv(k)</c> (states <c>{group}.k_conv_state</c> / <c>v_conv_state</c>).</item>
/// <item>A query-conditioned relative position bias: <c>attention.relative.weight</c> [heads·R, hidden] projects the layer input to R values
/// per head, <c>attention.relative_bias.weight</c> [E, R] maps them to one bias per backward distance 0 ≤ d &lt; E (CachedAttention
/// <c>relative_bias</c>).</item>
/// <item>A log-floor query temperature on the layers without a sliding window: <c>attention.log_scaling_floor</c> (&gt; 0) and
/// <c>attention.log_scaling_alpha</c> (see <see cref="AttentionTemperature.LogFloor"/>); it scales the queries and the relative bias.</item>
/// </list>
/// </remarks>
internal sealed class HybridAttention
{
    private readonly int _heads;
    private readonly int _kvHeads;
    private readonly int _headDim;
    private readonly float _scale;
    private readonly bool _rope;
    private readonly int _ropeDims;
    private readonly float _ropeBase;
    private readonly string _ropeStyle;
    private readonly float _eps;
    private readonly int _window;
    private readonly int[] _slidingLayers;
    private readonly float _localBase;
    private readonly int[] _layerHeads;
    private readonly int[] _layerKvHeads;
    private readonly bool _scaleByHeadDim;
    private readonly float _logScalingFloor;
    private readonly float _logScalingAlpha;

    /// <summary>Query heads.</summary>
    public int Heads => _heads;

    /// <summary>Key / value heads.</summary>
    public int KvHeads => _kvHeads;

    /// <summary>Width of one head.</summary>
    public int HeadDim => _headDim;

    /// <summary>Reads the <c>attention.*</c> and <c>rope.*</c> hyper-parameters of a source.</summary>
    public HybridAttention(ModelSource src)
    {
        _heads = src.Arch("attention.head_count", 0);
        _kvHeads = src.Arch("attention.kv_head_count", _heads);
        _headDim = src.Arch("attention.head_dim", 0);
        _scale = src.Arch("attention.scale", _headDim > 0 ? 1f / MathF.Sqrt(_headDim) : 1f);
        _rope = src.Arch("attention.rope", false);
        _ropeDims = src.Arch("rope.dim", _headDim);
        _ropeBase = src.Arch("rope.base", 10000f);
        _ropeStyle = src.Arch("rope.style", "neox");
        _eps = src.Arch("norm_eps", 1e-5f);
        _window = src.Arch("attention.sliding_window", 0);
        _slidingLayers = src.Arch("attention.sliding_layers", Array.Empty<int>());
        _localBase = src.Arch("rope.local_base", _ropeBase);
        _layerHeads = src.Arch("attention.layer_head_counts", Array.Empty<int>());
        _layerKvHeads = src.Arch("attention.layer_kv_head_counts", Array.Empty<int>());
        _scaleByHeadDim = src.Arch("attention.scale_by_head_dim", false);
        _logScalingFloor = src.Arch("attention.log_scaling_floor", 0f);
        _logScalingAlpha = src.Arch("attention.log_scaling_alpha", 0f);
    }

    private bool Sliding(int il) => _window > 0 && il < _slidingLayers.Length && _slidingLayers[il] != 0;

    /// <summary>Query heads of layer <paramref name="il"/> (the global count unless <c>attention.layer_head_counts</c> sets one).</summary>
    public int HeadsOf(int il) => il >= 0 && il < _layerHeads.Length && _layerHeads[il] > 0 ? _layerHeads[il] : _heads;

    /// <summary>Key / value heads of layer <paramref name="il"/> (the global count unless <c>attention.layer_kv_head_counts</c> sets one).</summary>
    public int KvHeadsOf(int il) => il >= 0 && il < _layerKvHeads.Length && _layerKvHeads[il] > 0 ? _layerKvHeads[il] : _kvHeads;

    /// <summary>Declares the key and value caches of one layer.</summary>
    public (StateDefinition Keys, StateDefinition Values) DefineStates(string group, DataType cacheType) => DefineStates(group, cacheType, -1);

    /// <summary>Declares the key and value caches of layer <paramref name="il"/> (with its own key / value head count).</summary>
    public (StateDefinition Keys, StateDefinition Values) DefineStates(string group, DataType cacheType, int il)
    {
        var shape = new Shape(Dim.Sym(WellKnown.ContextSymbol), KvHeadsOf(il), _headDim);
        cacheType = AdapterOptions.KvRowType(cacheType, _headDim);
        var keys = new StateDefinition { Name = $"{group}.k_cache", Type = cacheType, Shape = shape, Kind = StateKind.KeyCache, Group = group };
        var values = new StateDefinition { Name = $"{group}.v_cache", Type = cacheType, Shape = shape, Kind = StateKind.ValueCache, Group = group };
        return (keys, values);
    }

    /// <summary>
    /// Declares the window states of the residual key / value convolutions of a layer (<c>attention.k_conv.weight</c>,
    /// <c>attention.v_conv.weight</c>); null for a convolution the layer does not have.
    /// </summary>
    public static (StateDefinition? Keys, StateDefinition? Values) DefineConvStates(ModelSource src, string group, string p)
    {
        static StateDefinition? Window(ModelSource src, string group, string name, string weightName)
            => src.Weight(weightName) is { } w
                ? RecurrentStates.ConvWindow(group, name, (int)w.Shape[1].Fixed - 1, (int)w.Shape[0].Fixed)
                : null;

        return (Window(src, group, "k_conv_state", p + "attention.k_conv.weight"), Window(src, group, "v_conv_state", p + "attention.v_conv.weight"));
    }

    /// <summary>Emits the mixer of layer <paramref name="il"/> on the normalized input <paramref name="h"/> [T, hidden].</summary>
    /// <param name="b">The graph builder.</param>
    /// <param name="h">The normalized layer input [T, hidden].</param>
    /// <param name="positions">Positions [T, 1] (i32).</param>
    /// <param name="il">The layer index.</param>
    /// <param name="p">The layer's weight-name prefix.</param>
    /// <param name="keyCache">The layer's key cache.</param>
    /// <param name="valueCache">The layer's value cache.</param>
    /// <param name="keyConv">The window state of the residual key convolution, when the layer has one (<see cref="DefineConvStates"/>).</param>
    /// <param name="valueConv">The window state of the residual value convolution, when the layer has one.</param>
    public Value Build(GraphBuilder b, Value h, Value positions, int il, string p, StateDefinition keyCache, StateDefinition valueCache,
        StateDefinition? keyConv = null, StateDefinition? valueConv = null)
    {
        var q = b.Linear(h, p + "attention.q.weight");
        var k = ResidualConv(b, b.Linear(h, p + "attention.k.weight"), p + "attention.k_conv.weight", keyConv);
        var v = ResidualConv(b, b.Linear(h, p + "attention.v.weight"), p + "attention.v_conv.weight", valueConv);
        return Attend(b, h, q, k, v, positions, il, p, keyCache, valueCache);
    }

    /// <summary><c>x + causal_conv(x)</c> with the depthwise kernel <paramref name="weightName"/> [C, K] (no bias, no activation), when given.</summary>
    internal static Value ResidualConv(GraphBuilder b, Value x, string weightName, StateDefinition? window)
    {
        if (window is null)
        {
            return x;
        }

        var state = b.State(window);
        var conv = b.Emit("CausalConv1d", [x, b.Weight(weightName), state], Attributes.From(("activation", "none")), writes: [state])[0];
        return b.Add(x, conv);
    }

    /// <summary>
    /// Emits the mixer of layer <paramref name="il"/> from already projected queries [T, heads·head_dim] and keys / values [T, kv_heads·head_dim]
    /// (for layouts that project them differently, e.g. fused or adapted): the optional q/k norms, rotation, cache, attention, output gate (a
    /// projection of <paramref name="h"/>) and output projection.
    /// </summary>
    public Value Attend(GraphBuilder b, Value h, Value q, Value k, Value v, Value positions, int il, string p, StateDefinition keyCache, StateDefinition valueCache)
    {
        int heads = HeadsOf(il);
        int kvHeads = KvHeadsOf(il);
        q = b.Reshape(q, 0, heads, _headDim);
        k = b.Reshape(k, 0, kvHeads, _headDim);
        v = b.Reshape(v, 0, kvHeads, _headDim);
        if (b.TryWeight(p + "attention.q_norm.weight") is { } qNorm)
        {
            q = HeadNorm(b, q, qNorm);
        }

        if (b.TryWeight(p + "attention.k_norm.weight") is { } kNorm)
        {
            k = HeadNorm(b, k, kNorm);
        }

        bool sliding = Sliding(il);
        if (_rope)
        {
            var rope = Attributes.From(("dims", _ropeDims), ("base", sliding ? _localBase : _ropeBase), ("style", _ropeStyle));
            Value[] RopeInputs(Value t) => !sliding && b.TryWeight("rope.freq_factors") is { } factors ? [t, positions, factors] : [t, positions];
            q = b.Emit1("Rope", RopeInputs(q), rope);
            k = b.Emit1("Rope", RopeInputs(k), rope);
        }

        Value? temperature = !sliding && _logScalingFloor > 0 ? AttentionTemperature.LogFloor(b, positions, _logScalingAlpha, _logScalingFloor) : null;
        if (temperature is not null)
        {
            q = b.Mul(q, temperature);
        }

        b.CacheWrite(b.State(keyCache), k);
        b.CacheWrite(b.State(valueCache), v);
        var attrs = new List<(string, object)> { ("scale", _scaleByHeadDim ? 1f / _headDim : _scale), ("offset", WellKnown.PastScalar) };
        if (sliding)
        {
            attrs.Add(("window", _window));
        }

        Value[] inputs = [q, b.State(keyCache), b.State(valueCache)];
        if (b.TryWeight(p + "attention.relative_bias.weight") is { } relativeBias)
        {
            // Per head R relative values of the layer input, mixed into one bias per backward distance: [T, heads, E].
            long r = relativeBias.Shape[1].Fixed;
            var relative = b.MatMul(b.Reshape(b.Linear(h, p + "attention.relative.weight"), 0, heads, r), relativeBias, $"{p}relative_bias");
            if (temperature is not null)
            {
                relative = b.Mul(relative, temperature);
            }

            inputs = [.. inputs, relative];
            attrs.Add(("relative_bias", (int)relativeBias.Shape[0].Fixed));
        }

        b.RuntimeScalar(WellKnown.PastScalar);
        var attn = b.Emit1("CachedAttention", inputs, Attributes.From([.. attrs]), $"{p}attn");
        if (b.TryWeight(p + "attention.gate.weight") is { } gateWeight)
        {
            var gate = b.Sigmoid(b.Linear(h, p + "attention.gate.weight"));
            attn = b.Mul(attn, gateWeight.Shape[0].Fixed == heads ? b.Reshape(gate, 0, heads, 1) : b.Reshape(gate, 0, heads, _headDim));
        }

        return b.Linear(b.Reshape(attn, 0, heads * _headDim), p + "attention.o.weight", name: $"{p}attn_out");
    }

    // A [head_dim] weight is shared by the heads; a [heads, head_dim] weight (PLaMo-2) scales every head by its own.
    private Value HeadNorm(GraphBuilder b, Value x, Value weight)
        => weight.Shape.Rank == 1 ? b.RmsNorm(x, weight, _eps) : b.Mul(b.RmsNorm(x, null, _eps), weight);
}
