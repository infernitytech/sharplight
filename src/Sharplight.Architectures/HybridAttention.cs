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
    }

    private bool Sliding(int il) => _window > 0 && il < _slidingLayers.Length && _slidingLayers[il] != 0;

    /// <summary>Declares the key and value caches of one layer.</summary>
    public (StateDefinition Keys, StateDefinition Values) DefineStates(string group, DataType cacheType)
    {
        var shape = new Shape(Dim.Sym(WellKnown.ContextSymbol), _kvHeads, _headDim);
        cacheType = AdapterOptions.KvRowType(cacheType, _headDim);
        var keys = new StateDefinition { Name = $"{group}.k_cache", Type = cacheType, Shape = shape, Kind = StateKind.KeyCache, Group = group };
        var values = new StateDefinition { Name = $"{group}.v_cache", Type = cacheType, Shape = shape, Kind = StateKind.ValueCache, Group = group };
        return (keys, values);
    }

    /// <summary>Emits the mixer of layer <paramref name="il"/> on the normalized input <paramref name="h"/> [T, hidden].</summary>
    public Value Build(GraphBuilder b, Value h, Value positions, int il, string p, StateDefinition keyCache, StateDefinition valueCache)
    {
        var q = b.Reshape(b.Linear(h, p + "attention.q.weight"), 0, _heads, _headDim);
        var k = b.Reshape(b.Linear(h, p + "attention.k.weight"), 0, _kvHeads, _headDim);
        var v = b.Reshape(b.Linear(h, p + "attention.v.weight"), 0, _kvHeads, _headDim);
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

        b.CacheWrite(b.State(keyCache), k);
        b.CacheWrite(b.State(valueCache), v);
        var attrs = new List<(string, object)> { ("scale", _scale), ("offset", WellKnown.PastScalar) };
        if (sliding)
        {
            attrs.Add(("window", _window));
        }

        b.RuntimeScalar(WellKnown.PastScalar);
        var attn = b.Emit1("CachedAttention", [q, b.State(keyCache), b.State(valueCache)], Attributes.From([.. attrs]), $"{p}attn");
        if (b.TryWeight(p + "attention.gate.weight") is { } gateWeight)
        {
            var gate = b.Sigmoid(b.Linear(h, p + "attention.gate.weight"));
            attn = b.Mul(attn, gateWeight.Shape[0].Fixed == _heads ? b.Reshape(gate, 0, _heads, 1) : b.Reshape(gate, 0, _heads, _headDim));
        }

        return b.Linear(b.Reshape(attn, 0, _heads * _headDim), p + "attention.o.weight", name: $"{p}attn_out");
    }

    // A [head_dim] weight is shared by the heads; a [heads, head_dim] weight (PLaMo-2) scales every head by its own.
    private Value HeadNorm(GraphBuilder b, Value x, Value weight)
        => weight.Shape.Rank == 1 ? b.RmsNorm(x, weight, _eps) : b.Mul(b.RmsNorm(x, null, _eps), weight);
}
