using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

/// <summary>
/// Registry of the ops whose kernels are generated per node from their attributes: linear attention, deformable convolution, Unique,
/// TF-IDF n-gram counting and the classical machine-learning ops (decision forests, one-vs-one voting, numeric label encoding) and long DFTs.
/// </summary>
public sealed partial class GraphLowerer
{
    private static string[] ExtendedOpNames => ["LinearAttention", "DeformConv", "Unique", "TfIdf", "TreeEnsemble", "OneVsOne", "LabelEncode", "Dft"];
    private static readonly HashSet<string> s_extendedOps = [.. ExtendedOpNames];

    /// <summary>Largest flattened Unique input on the device: the kernels compare every pair of elements.</summary>
    private const long MaxUniqueElements = 1 << 16;

    /// <summary>Rows assumed for a symbolic outer axis of a Unique input when checking <see cref="MaxUniqueElements"/>.</summary>
    private const long MaxUniqueRowsPerSymbol = 1024;

    private static OpSupport SupportsExtended(Node node)
    {
        string op = node.Op.Name;
        switch (op)
        {
            case "LinearAttention":
                if (node.Inputs.Any(v => !IsFloatType(v.Type)))
                {
                    return OpSupport.No("LinearAttention needs float inputs");
                }

                return node.Inputs.Any(v => v.Shape.Dims.Skip(1).Any(d => d.IsSymbolic))
                    ? OpSupport.No("LinearAttention with a symbolic inner axis")
                    : OpSupport.Yes;
            case "DeformConv":
                if (node.Inputs.Any(v => !IsFloatType(v.Type)) || node.Inputs[0].Rank != 4)
                {
                    return OpSupport.No("DeformConv needs float 4-D inputs");
                }

                return OpSupport.Yes;
            case "Unique":
                var x = node.Inputs[0];
                if (x.Type.IsQuantized || !(IsFloatType(x.Type) || x.Type == DataType.I32 || x.Type == DataType.I64 || x.Type == DataType.U32))
                {
                    return OpSupport.No($"Unique of {x.Type}");
                }

                // The kernels cost (live elements)²: a symbolic input is taken when its rows are short (the bound is not known here).
                long inner = x.Shape.Dims.Skip(1).Aggregate(1L, (s, d) => d.IsSymbolic ? long.MaxValue / 2 : s * d.Fixed);
                long elements = x.Shape.Rank == 0 ? 1 : x.Shape[0].IsSymbolic ? inner * MaxUniqueRowsPerSymbol : inner * x.Shape[0].Fixed;
                return elements <= MaxUniqueElements ? OpSupport.Yes : OpSupport.No($"Unique of more than {MaxUniqueElements} elements");
            case "TfIdf":
                return node.Inputs[0].Type == DataType.I32 && node.Inputs[0].Shape.Dims.Skip(1).All(d => !d.IsSymbolic)
                    ? OpSupport.Yes
                    : OpSupport.No("TfIdf needs i32 token ids with a fixed row length");
            case "Dft":
                return IsFloatType(node.Inputs[0].Type) && node.Attributes.Get<long>("length") <= MaxDeviceDftLength && node.Inputs[0].Shape.Dims.Skip(1).All(d => !d.IsSymbolic)
                    ? OpSupport.Yes
                    : OpSupport.No($"Dft longer than {MaxDeviceDftLength} or with a symbolic inner axis");
            case "TreeEnsemble":
            case "OneVsOne":
                return IsFloatType(node.Inputs[0].Type) && !node.Inputs[0].Shape[1].IsSymbolic ? OpSupport.Yes : OpSupport.No($"{op} needs a float [N, F] input");
            default:
                return LabelEncodeTable(node) is not null ? OpSupport.Yes : OpSupport.No("LabelEncode of strings");
        }
    }

    private bool LowerExtended(Node n)
    {
        switch (n.Op.Name)
        {
            case "LinearAttention": LowerLinearAttention(n); return true;
            case "DeformConv": LowerDeformConv(n); return true;
            case "Unique": LowerUnique(n); return true;
            case "TfIdf": LowerTfIdf(n); return true;
            case "TreeEnsemble": LowerTreeEnsemble(n); return true;
            case "OneVsOne": LowerOneVsOne(n); return true;
            case "LabelEncode": LowerLabelEncode(n); return true;
            case "Dft": LowerDft(n); return true;
        }

        return false;
    }

    /// <summary>A read-only device table of 32-bit words (at least one word), uploaded with the weights.</summary>
    private TensorRef ConstTable(string name, uint[] words)
    {
        if (words.Length == 0)
        {
            words = [0];
        }

        byte[] bytes = MemoryMarshal.AsBytes(words.AsSpan()).ToArray();
        var buffer = NewBuffer(BufferKind.Weight, "const:" + name, bytes.Length, init: bytes);
        return new TensorRef(buffer, 0, DataType.U32, [words.Length], [1], null);
    }
}
