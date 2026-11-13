using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Cpu;

// Support checks for gather/scatter and other index-driven ops.
internal static partial class CpuOpSupport
{
    private static OpSupport CheckGatherFamily(Node node)
    {
        var ins = node.Inputs;
        string op = node.Op.Name;
        if (!Plain(ins[0]))
        {
            return OpSupport.No($"{op} data type {ins[0].Type} is not supported");
        }

        if (!IsIndex(ins[1]))
        {
            return OpSupport.No($"{op} indices must be i32 or i64, got {ins[1].Type}");
        }

        return OpSupport.Yes;
    }

    private static OpSupport CheckScatter(Node node)
    {
        var ins = node.Inputs;
        string op = node.Op.Name;
        var data = ins[0].Type;
        if (!(data == DataType.F32 || data == DataType.I32 || data == DataType.I64))
        {
            return OpSupport.No($"{op} data must be f32, i32 or i64, got {data}");
        }

        if (!IsIndex(ins[1]))
        {
            return OpSupport.No($"{op} indices must be i32 or i64, got {ins[1].Type}");
        }

        if (ins[2].Type != data)
        {
            return OpSupport.No($"{op} updates must have the data type");
        }

        string reduction = node.Attributes.Get("reduction", "none");
        return reduction is "none" or "add" or "mul" or "max" or "min"
            ? OpSupport.Yes
            : OpSupport.No($"{op} reduction '{reduction}' is not supported");
    }

    private static OpSupport CheckTrilu(Node node)
    {
        var input = node.Inputs[0];
        return Plain(input) && input.Rank >= 2 ? OpSupport.Yes : OpSupport.No("Trilu needs rank >= 2 of a plain type");
    }

    private static OpSupport CheckOneHot(Node node)
    {
        var indices = node.Inputs[0];
        return IsIndex(indices) ? OpSupport.Yes : OpSupport.No($"OneHot indices must be i32 or i64, got {indices.Type}");
    }

    private static OpSupport CheckSequencePool(Node node)
    {
        var ins = node.Inputs;
        if (ins[0].Type != DataType.F32)
        {
            return OpSupport.No("SequencePool input must be f32");
        }

        if (ins[1].Type != DataType.I32 || ins[1].Rank != 2 || ins[1].Shape[1] != 2)
        {
            return OpSupport.No("SequencePool spans must be i32 [B, 2]");
        }

        if (node.Attributes.Get("mode", "mean") is not ("mean" or "sum" or "max" or "first" or "last"))
        {
            return OpSupport.No("SequencePool mode must be mean | sum | max | first | last");
        }

        return OpSupport.Yes;
    }
}
