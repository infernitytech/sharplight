using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Cpu;

// Integer elementwise arithmetic, Narrow and Reinterpret (StandardOps.Numeric).
internal static partial class CpuOpSupport
{
    private static readonly HashSet<string> s_integerOps = Enum.GetNames<IntOpKind>().ToHashSet(StringComparer.Ordinal);

    /// <summary>An elementwise op whose inputs are all i32: computed exactly by <see cref="IntElementwiseOp"/>.</summary>
    internal static bool IsIntegerElementwise(Node node) => s_integerOps.Contains(node.Op.Name) && node.Inputs.All(i => i.Type == DataType.I32);

    private static OpSupport CheckNarrow(Node node)
    {
        var format = NumericFormats.Parse(node.Attributes.Get<string>("format"));
        var type = node.Inputs[0].Type;
        return type == DataType.F32 || (type == DataType.I32 && !NumericFormats.IsFloat(format))
            ? OpSupport.Yes
            : OpSupport.No($"Narrow of {type} to {format}");
    }
}
