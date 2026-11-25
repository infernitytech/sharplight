using System.Globalization;
using System.Text;
using Sharplight.Graph;

namespace Sharplight.Shaders.Lowering;

public sealed partial class GraphLowerer
{
    /// <summary>
    /// The generated code of a matmul store epilogue: the <c>mm_epi(m0, a0, a1, a2)</c> source, the tensors it reads with their
    /// access kinds (see <see cref="MatMulEpilogueArgKind"/>), the tensor it writes and the fused group's label.
    /// </summary>
    private sealed record MatMulEpilogueCode(string Source, TensorRef[] Args, int[] Kinds, TensorRef Output, List<Node> Group, string Label, Value OutputValue);

    /// <summary>
    /// The epilogue code of <paramref name="epilogue"/> applied to matmul <paramref name="matmul"/>'s product, for the kernels of
    /// <paramref name="kernelFile"/> (matmul.slang or mmvq.slang).
    /// </summary>
    private MatMulEpilogueCode MatMulEpilogueSource(Node matmul, MatMulEpilogue epilogue, string kernelFile = "matmul.slang")
    {
        var name = new Dictionary<Value, string> { [matmul.Output] = "m0" };
        var inGroup = epilogue.Group.Select(n => n.Output).ToHashSet();
        var args = new List<TensorRef>();
        var kinds = new List<int>();
        var columns = matmul.Inputs[1].Shape[0];
        foreach (var v in epilogue.Group.SelectMany(n => n.Inputs).Where(v => v != matmul.Output && !inGroup.Contains(v)).Distinct())
        {
            name[v] = "a" + args.Count;
            kinds.Add(MatMulEpilogueArgKind(v, matmul.Output.Shape, columns));
            args.Add(AsDenseF32(Ref(v), "matmul-epilogue"));
        }

        var src = new StringBuilder();
        src.AppendLine("#include \"math_ext.slang\"");
        src.AppendLine("float mm_epi(float m0, float a0, float a1, float a2) {");
        foreach (var node in epilogue.Group)
        {
            var local = "v" + name.Count;
            src.AppendLine($"  float {local} = {OpExpr(node, [.. node.Inputs.Select(i => name[i])])};");
            name[node.Output] = local;
        }

        src.AppendLine($"  return {name[epilogue.Output]};");
        src.AppendLine("}");
        src.AppendLine($"#include \"{kernelFile}\"");
        string label = $" fused[{string.Join(",", epilogue.Group.Select(g => g.Op.Name))}]";
        return new MatMulEpilogueCode(src.ToString(), [.. args], [.. kinds], Produce(epilogue.Output), epilogue.Group, label, epilogue.Output);
    }

    /// <summary>
    /// A kernel compiled with the store epilogue <paramref name="code"/> (MM_EPI, mm_epi.slang): of matmul.slang, or of mmvq.slang for
    /// <paramref name="file"/> <c>mmvq_epi.slang</c> (the source must include the matching kernel file).
    /// </summary>
    private CompiledKernel MatMulEpilogueKernel(MatMulEpilogueCode code, string entry, Dictionary<string, string> macros, string file = "matmul_epi.slang")
    {
        var m = new Dictionary<string, string>(macros)
        {
            ["MM_EPI"] = "1",
            ["KERNEL_" + entry.ToUpperInvariant()] = "1",
            ["MM_EPI_ARGS"] = code.Args.Length.ToString(CultureInfo.InvariantCulture),
        };
        for (int i = 0; i < code.Kinds.Length; i++)
        {
            m["MM_EPI_K" + i] = code.Kinds[i].ToString(CultureInfo.InvariantCulture);
        }

        return _compiler.Compile(file, code.Source, entry, _opt.Target, m, _opt.Cache);
    }

    /// <summary>The epilogue's tensor arguments, in the order of the kernel parameters <c>e0</c> .. <c>e2</c>.</summary>
    private static KernelArg[] MatMulEpilogueArgs(MatMulEpilogueCode? code) => code is null ? [] : [.. code.Args.Select(a => Pointer(a))];

    /// <summary>
    /// Lowers matmul <paramref name="matmul"/> with the elementwise group <paramref name="epilogue"/> consuming its product: in
    /// the split-f16 GEMM's or the int-dot matvec's store where those serve the matmul, else as the matmul followed by the group's
    /// own launch.
    /// </summary>
    private void LowerMatMulEpilogue(Node matmul, MatMulEpilogue epilogue) => LowerMatMul(matmul, epilogue);
}
