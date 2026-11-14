using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Cpu;

// Support checks for transformer, recurrent and state-space ops.
internal static partial class CpuOpSupport
{
    private static OpSupport CheckRope(Node node)
    {
        var ins = node.Inputs;
        if (ins[0].Type != DataType.F32 && !IsStatic(ins[0]))
        {
            return OpSupport.No("Rope input must be f32");
        }

        if (ins[0].Rank != 3)
        {
            return OpSupport.No("Rope input must be [T, H, D]");
        }

        if (ins[1].Type != DataType.I32)
        {
            return OpSupport.No("Rope positions must be i32");
        }

        return OpSupport.Yes;
    }

    private static OpSupport CheckCacheWrite(Node node)
    {
        var ins = node.Inputs;
        var cacheType = ins[0].Type;
        if (!(cacheType == DataType.F32 || cacheType == DataType.F16 || cacheType == DataType.BF16 || IsQuantizedKvCache(ins[0])))
        {
            return OpSupport.No($"KV cache type {cacheType} is not supported by the CPU provider");
        }

        return ins[1].Type == DataType.F32 ? OpSupport.Yes : OpSupport.No("CacheWrite source must be f32");
    }

    private static OpSupport CheckCachedAttention(Node node)
    {
        var ins = node.Inputs;
        if (ins[0].Type != DataType.F32)
        {
            return OpSupport.No("CachedAttention q must be f32");
        }

        if (node.Attributes.Get("sinks", false) && ins[3].Type != DataType.F32)
        {
            return OpSupport.No("CachedAttention sinks must be f32");
        }

        if (!(IsKvCacheType(ins[1].Type) || IsQuantizedKvCache(ins[1])) || !(IsKvCacheType(ins[2].Type) || IsQuantizedKvCache(ins[2])))
        {
            return OpSupport.No($"KV cache type {ins[1].Type}/{ins[2].Type} is not supported by the CPU provider (f32, f16, q8_0, q4_0)");
        }

        int selectionInput = node.Attributes.Get("sinks", false) ? 4 : 3;
        if (node.Attributes.Get("selection", false)
            && (ins.Length <= selectionInput || ins[selectionInput].Type != DataType.I32 || node.Attributes.Get("window", 0) > 0 || node.Attributes.Get("chunk", 0) > 0))
        {
            return OpSupport.No("CachedAttention selection must be i32 [T, K] and cannot be combined with window or chunk");
        }

        return OpSupport.Yes;
    }

    private static OpSupport CheckIndexerTopK(Node node)
    {
        var ins = node.Inputs;
        var selection = IndexerSelection.From(node.Attributes);
        if (ins[0].Type != DataType.F32 || (!selection.PerHead && ins[2].Type != DataType.F32))
        {
            return OpSupport.No("IndexerTopK q and weights must be f32");
        }

        return IsKvCacheType(ins[1].Type) ? OpSupport.Yes : OpSupport.No($"IndexerTopK key cache type {ins[1].Type} (f32 or f16)");
    }

    private static OpSupport CheckAttention(Node node)
    {
        var ins = node.Inputs;
        if (ins[0].Type != DataType.F32)
        {
            return OpSupport.No("Attention q must be f32");
        }

        if (!IsKvCacheType(ins[1].Type) || !IsKvCacheType(ins[2].Type))
        {
            return OpSupport.No("Attention k/v must be f32 or f16");
        }

        if (ins.Length > 3)
        {
            bool isMask = ins[3].Type == DataType.F32 && ins[3].Rank == 2;
            bool isSpans = ins[3].Type == DataType.I32 && ins[3].Rank == 2 && ins[3].Shape[1] == 2;
            if (!isMask && !isSpans)
            {
                return OpSupport.No("Attention fourth input must be an f32 [T, S] mask or i32 [T, 2] spans");
            }
        }

        return OpSupport.Yes;
    }

    private static OpSupport CheckCausalConv1d(Node node)
    {
        var ins = node.Inputs;
        if (ins[0].Type != DataType.F32 || ins[2].Type != DataType.F32)
        {
            return OpSupport.No("CausalConv1d needs f32 activations and state");
        }

        bool hasBias = node.Attributes.Get("bias", false) || (ins.Length > 3 && ins[3].Type != DataType.I32);
        return F32Like(ins[1]) && (!hasBias || F32Like(ins[3]))
            ? OpSupport.Yes
            : OpSupport.No("CausalConv1d weight/bias must be f32-convertible");
    }

    private static OpSupport CheckSelectiveScan(Node node)
    {
        var ins = node.Inputs;
        if (ins[0].Type != DataType.F32 || ins[1].Type != DataType.F32 || ins[4].Type != DataType.F32 || ins[5].Type != DataType.F32 || ins[7].Type != DataType.F32)
        {
            return OpSupport.No("SelectiveScan needs f32 x/dt/B/C and state");
        }

        if (ins.Length > 8 && ins[8].Type != DataType.I32)
        {
            return OpSupport.No("SelectiveScan sequence runs must be i32");
        }

        return F32Like(ins[2]) && F32Like(ins[3]) && F32Like(ins[6])
            ? OpSupport.Yes
            : OpSupport.No("SelectiveScan dt_bias/A/D must be f32-convertible");
    }

    private static OpSupport CheckGatedDeltaRule(Node node)
    {
        var ins = node.Inputs;
        foreach (var v in ins.Take(5))
        {
            if (v.Type != DataType.F32)
            {
                return OpSupport.No($"GatedDeltaRule needs f32 inputs, got {v.Type}");
            }
        }

        if (ins[5].Type != DataType.F32 && ins[5].Type != DataType.BF16 && ins[5].Type != DataType.F16)
        {
            return OpSupport.No($"GatedDeltaRule state must be f32, bf16 or f16, got {ins[5].Type}");
        }

        if (ins.Length > 6 && ins[6].Type != DataType.I32)
        {
            return OpSupport.No("GatedDeltaRule sequence runs must be i32");
        }

        return node.Attributes.Get("rule", "delta") is "delta" or "additive" ? OpSupport.Yes : OpSupport.No("GatedDeltaRule rule must be delta or additive");
    }

    /// <summary>RWKV recurrences: f32 activations and state; the optional trailing sequence-runs table is i32 (<paramref name="inputs"/> counts the inputs before it).</summary>
    private static OpSupport CheckWkv(Node node, int inputs)
    {
        for (int i = 0; i < inputs; i++)
        {
            if (node.Inputs[i].Type != DataType.F32 && !(i == 4 && inputs == 6 && F32Like(node.Inputs[i])))
            {
                return OpSupport.No($"{node.Op.Name} needs f32 inputs and state, got {node.Inputs[i].Type}");
            }
        }

        return node.Inputs.Length > inputs && node.Inputs[inputs].Type != DataType.I32 ? OpSupport.No($"{node.Op.Name} sequence runs must be i32") : OpSupport.Yes;
    }
}
