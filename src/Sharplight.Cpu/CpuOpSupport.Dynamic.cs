using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Cpu;

// Ops with symbolic or data-dependent output sizes.
internal static partial class CpuOpSupport
{
    private static OpSupport CheckFill(Node node)
    {
        var type = node.Output.Type;
        return type == DataType.F32 || type == DataType.I32 || type == DataType.I64 ? OpSupport.Yes : OpSupport.No($"Fill of {type}");
    }

    private static OpSupport CheckDataDependentInput(Node node)
    {
        var type = node.Inputs[0].Type;
        bool readable = type == DataType.F32 || type == DataType.I32 || type == DataType.I64 || type == DataType.I8 || type == DataType.U8 || type == DataType.Bool;
        return readable ? OpSupport.Yes : OpSupport.No($"{node.Op.Name} on {type}");
    }
}
