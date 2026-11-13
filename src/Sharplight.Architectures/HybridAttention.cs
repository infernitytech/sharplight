using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Grouped-query attention mixer of the <see cref="HybridAdapter"/> over a KV cache: weights <c>attention.{q,k,v,o}.weight</c>, optional per-head q/k RMS norms (<c>attention.{q,k}_norm.weight</c> [head_dim]), optional rotary
/// embeddings (<c>attention.rope</c>), scale <c>attention.scale</c> (default 1/sqrt(head_dim)).
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
    }

    /// <summary>Declares the key and value caches of one layer.</summary>
    public (StateDefinition Keys, StateDefinition Values) DefineStates(string group, DataType cacheType)
    {
        var shape = new Shape(Dim.Sym(WellKnown.ContextSymbol), _kvHeads, _headDim);
        var keys = new StateDefinition { Name = $"{group}.k_cache", Type = cacheType, Shape = shape, Kind = StateKind.KeyCache, Group = group };
        var values = new StateDefinition { Name = $"{group}.v_cache", Type = cacheType, Shape = shape, Kind = StateKind.ValueCache, Group = group };
        return (keys, values);
    }

    /// <summary>Emits the mixer on the normalized input <paramref name="h"/> [T, hidden].</summary>
    public Value Build(GraphBuilder b, Value h, Value positions, string p, StateDefinition keyCache, StateDefinition valueCache)
    {
        var q = b.Reshape(b.Linear(h, p + "attention.q.weight"), 0, _heads, _headDim);
        var k = b.Reshape(b.Linear(h, p + "attention.k.weight"), 0, _kvHeads, _headDim);
        var v = b.Reshape(b.Linear(h, p + "attention.v.weight"), 0, _kvHeads, _headDim);
        if (b.TryWeight(p + "attention.q_norm.weight") is { } qNorm)
        {
            q = b.RmsNorm(q, qNorm, _eps);
        }

        if (b.TryWeight(p + "attention.k_norm.weight") is { } kNorm)
        {
            k = b.RmsNorm(k, kNorm, _eps);
        }

        if (_rope)
        {
            var rope = Attributes.From(("dims", _ropeDims), ("base", _ropeBase), ("style", _ropeStyle));
            q = b.Emit1("Rope", [q, positions], rope);
            k = b.Emit1("Rope", [k, positions], rope);
        }

        b.CacheWrite(b.State(keyCache), k);
        b.CacheWrite(b.State(valueCache), v);
        var attn = b.CachedAttention(q, b.State(keyCache), b.State(valueCache), _scale, name: $"{p}attn");
        return b.Linear(b.Reshape(attn, 0, _heads * _headDim), p + "attention.o.weight", name: $"{p}attn_out");
    }
}
