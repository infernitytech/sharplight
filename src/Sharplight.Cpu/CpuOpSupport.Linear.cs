using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Cpu;

// Support checks for matrix products and embedding lookups.
internal static partial class CpuOpSupport
{
    private static OpSupport CheckMatMul(Node node)
    {
        return CheckQuantizedMatMul(node, "MatMul", kDimension: 1, indexInput: null);
    }

    private static OpSupport CheckMatMulId(Node node)
    {
        return CheckQuantizedMatMul(node, "MatMulId", kDimension: 2, indexInput: node.Inputs[2]);
    }

    /// <summary>Shared by MatMul and MatMulId: f32 activations, a weight whose K is a whole number of blocks.</summary>
    private static OpSupport CheckQuantizedMatMul(Node node, string op, int kDimension, Value? indexInput)
    {
        var ins = node.Inputs;
        if (!F32Like(ins[0]))
        {
            return OpSupport.No($"{op} activations must be f32, got {ins[0].Type}");
        }

        var weight = ins[1];
        var format = Quant.FormatOf(weight.Type);
        if (format is null)
        {
            return OpSupport.No($"{op} weight type {weight.Type} is not supported by the CPU provider");
        }

        if (!IsStatic(weight) && weight.Type != DataType.F32)
        {
            return OpSupport.No($"{op} with a non-f32 activation as the weight");
        }

        if (indexInput is not null && indexInput.Type != DataType.I32)
        {
            return OpSupport.No("MatMulId expert indices must be i32");
        }

        long k = weight.Shape.Dims[kDimension].Fixed;
        if (k % Quant.BlockSize(format.Value) != 0)
        {
            return OpSupport.No($"K={k} is not a multiple of the {weight.Type} block size");
        }

        return OpSupport.Yes;
    }

    private static OpSupport CheckGather(Node node)
    {
        var ins = node.Inputs;
        var table = ins[0];
        if (Quant.FormatOf(table.Type) is null)
        {
            return OpSupport.No($"Gather table type {table.Type} is not supported by the CPU provider");
        }

        if (!IsStatic(table) && table.Type.IsQuantized)
        {
            return OpSupport.No("Gather from a quantized activation");
        }

        if (ins[1].Type != DataType.I32)
        {
            return OpSupport.No("Gather indices must be i32");
        }

        return OpSupport.Yes;
    }
}
