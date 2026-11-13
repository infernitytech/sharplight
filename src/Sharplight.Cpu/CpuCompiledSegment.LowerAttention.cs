using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Cpu;

// Transformer ops: rotary embedding, KV-cache writes and attention.
internal sealed unsafe partial class CpuCompiledSegment
{
    private void LowerRope(Node n)
    {
        var a = n.Attributes;
        var ins = n.Inputs;
        var x = RowContig(RefF32(ins[0]));
        var positions = Ref(ins[1]);
        TRef? frequencyFactors = ins.Length > 2 ? RefF32(ins[2]) : null;
        var y = NewDense(n.Output);
        var rope = new RopeOp(
            x, positions, frequencyFactors, y, a.Get("dims", 0), a.Get("base", 10000f), a.Get("style", "neox"),
            IntArray(a, "sections"), a.Get("interleaved", false), a.Get("freq_scale", 1f));
        AddStep(rope, Touched(x, positions, frequencyFactors, y));
        _refs[n.Output] = y;
    }

    /// <summary>Writes new K/V rows into the cache state; produces no value (the cache is a state tensor).</summary>
    private void LowerCacheWrite(Node n)
    {
        var a = n.Attributes;
        var ins = n.Inputs;
        var cache = Ref(ins[0]);
        var x = RowContig(RefF32(ins[1]));
        if (ins.Length > 2)
        {
            var rows = Dense(Ref(ins[2]));
            AddStep(new CacheWriteOp(cache, x, a.Get("offset", WellKnown.PastScalar), rows, (int)a.Get("slots", 1L)), cache, x, rows);
            return;
        }

        AddStep(new CacheWriteOp(cache, x, a.Get("offset", WellKnown.PastScalar)), cache, x);
    }

    private void LowerCachedAttention(Node n)
    {
        var a = n.Attributes;
        var ins = n.Inputs;
        var q = RowContig(Ref(ins[0]));
        var k = Ref(ins[1]);
        var v = Ref(ins[2]);
        var y = NewDense(n.Output);
        bool hasSinks = a.Get("sinks", false);
        int sinksOffset = hasSinks ? 1 : 0; // inputs: q, k, v, [sinks], [rows, runs]
        TRef? sinks = hasSinks ? Ref(ins[3]) : null;
        TRef? rows = ins.Length > 3 + sinksOffset ? Dense(Ref(ins[3 + sinksOffset])) : null;
        var attention = new AttentionOp(
            q, k, v, null, y, a.Get("scale", 1f), a.Get("offset", WellKnown.PastScalar), true, a.Get("window", 0), a.Get("softcap", 0f),
            rows, (int)a.Get("slots", 1L), sinks);
        AddStep(attention, Touched(q, k, v, sinks, rows, y));
        _refs[n.Output] = y;
    }

    /// <summary>Attention over full K/V; the optional fourth input is either an f32 mask or i32 per-row [start, end) spans.</summary>
    private void LowerAttention(Node n)
    {
        var a = n.Attributes;
        var ins = n.Inputs;
        var q = RowContig(Ref(ins[0]));
        var k = RowContig(Ref(ins[1]));
        var v = RowContig(Ref(ins[2]));
        bool hasSpans = ins.Length > 3 && ins[3].Type == DataType.I32;
        TRef? extra = ins.Length > 3 ? hasSpans ? Dense(Ref(ins[3])) : Ref(ins[3]) : null;
        var y = NewDense(n.Output);
        var attention = new AttentionOp(
            q, k, v, hasSpans ? null : extra, y, a.Get("scale", 1f), null, a.Get("causal", false), a.Get("window", 0), a.Get("softcap", 0f),
            spans: hasSpans ? extra : null);
        AddStep(attention, Touched(q, k, v, extra, y));
        _refs[n.Output] = y;
    }
}
