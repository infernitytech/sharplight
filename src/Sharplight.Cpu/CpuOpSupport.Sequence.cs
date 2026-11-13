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
        if (!(cacheType == DataType.F32 || cacheType == DataType.F16 || cacheType == DataType.BF16))
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

        if (!IsKvCacheType(ins[1].Type) || !IsKvCacheType(ins[2].Type))
        {
            return OpSupport.No($"KV cache type {ins[1].Type}/{ins[2].Type} is not supported by the CPU provider (f32/f16 only)");
        }

        return OpSupport.Yes;
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
        foreach (var v in ins.Take(6))
        {
            if (v.Type != DataType.F32)
            {
                return OpSupport.No($"GatedDeltaRule needs f32 inputs and state, got {v.Type}");
            }
        }

        if (ins.Length > 6 && ins[6].Type != DataType.I32)
        {
            return OpSupport.No("GatedDeltaRule sequence runs must be i32");
        }

        return OpSupport.Yes;
    }
}
