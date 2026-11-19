using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;
using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

public sealed partial class GraphLowerer
{
    /// <summary>Threads per block of <c>gemm_fp8_native.cu</c> (2 x 4 warps).</summary>
    private const int NativeFp8Threads = 256;

    /// <summary>Token rows and weight rows of a <c>gemm_fp8_native.cu</c> tile.</summary>
    private const int NativeFp8Tile = 128;

    /// <summary>Dynamic shared memory of <c>gemm_fp8_native.cu</c>: three stages of 128 + 128 rows of 128 codes and one scale each.</summary>
    private const int NativeFp8SharedBytes = 3 * (256 * 128 + 256 * 4);

    /// <summary>
    /// Fewest activation rows a matmul runs on a hand-written provider GEMM with (<c>SHARPLIGHT_NATIVE_GEMM_MIN_ROWS</c>, default
    /// 256): below that the generated GEMMs' 64-row tiles and K splits fill the GPU better.
    /// </summary>
    private static long NativeGemmMinRows => DiagnosticSwitches.NativeGemmMinRows ?? 256;

    /// <summary>
    /// True when matmuls of block-scaled E4M3 weights [<paramref name="n"/>, <paramref name="k"/>] run on the hand-written FP8 GEMM
    /// (<see cref="DeviceFeatures.NativeGemmFp8"/>, <c>gemm_fp8_native.cu</c>) from <see cref="NativeGemmMinRows"/> rows: weights in
    /// the split 128-value layout (the activation rows' format), 16-byte aligned rows (K a multiple of 512) and an even N (paired
    /// stores), on a target that takes CUDA C++.
    /// </summary>
    private bool UseNativeFp8Gemm(DataType type, long n, long k)
        => _opt.Has(DeviceFeatures.NativeGemmFp8) && type == ShaderTypes.F8E4M3_B128R && k % 512 == 0 && n % 2 == 0
           && _compiler.CompilesCuda(_opt.Target);

    /// <summary>The hand-written FP8 GEMM for [<paramref name="n"/>, <paramref name="k"/>] weights, single or dual with an epilogue.</summary>
    private CompiledKernel NativeFp8Kernel(long n, long k, string? dualEpilogue)
    {
        var macros = Macros(("K", k), ("N", n), ("Y_STRIDE", n), ("DUAL", dualEpilogue is null ? 0 : 1));
        return _compiler.CompileCuda("gemm_fp8_native.cu", dualEpilogue ?? "", "gemm_fp8_native", NativeFp8Threads, NativeFp8SharedBytes,
            _opt.Target, macros, _opt.Cache);
    }

    /// <summary>Grid of the hand-written FP8 GEMM: token tiles in x, weight tiles in y (a dual block covers half a tile of outputs).</summary>
    private static Grid NativeFp8Grid(long rows, long n, bool dual)
        => new(Grid.Div(rows, NativeFp8Tile), Grid.Div(n, dual ? NativeFp8Tile / 2 : NativeFp8Tile));

    /// <summary>Threads per block of <c>gemm_f16_native.cu</c> per tile configuration (0: 128 x 256, 1: 128 x 128).</summary>
    private static readonly int[] s_nativeF16Threads = [256, 128];

    /// <summary>Weight rows per tile of <c>gemm_f16_native.cu</c> per tile configuration.</summary>
    private static readonly int[] s_nativeF16TileColumns = [256, 128];

    /// <summary>Dynamic shared memory of <c>gemm_f16_native.cu</c> per tile configuration: two stages of 384 rows, three of 256 rows.</summary>
    private static readonly int[] s_nativeF16Shared = [2 * 384 * 128, 3 * 256 * 128];

    /// <summary>
    /// Relative speed of the f16 tile configurations at full waves (RTX PRO 4500 Blackwell, Qwen3-8B shapes at 2048 rows: 128 x 256
    /// 155-161 TFLOPS, 128 x 128 140-150).
    /// </summary>
    private static readonly double[] s_nativeF16Speed = [1.0, 0.94];

    /// <summary>
    /// True when the f16 products of the library GEMM path (bf16 weights stored as f16, the f16 copies of x) run on the hand-written
    /// f16 GEMM of the provider (<see cref="DeviceFeatures.NativeGemmF16"/>, <c>gemm_f16_native.cu</c>), preferred over the vendor
    /// library when both are there; disabling <c>native_gemm_f16</c> leaves them to the library (or the generated GEMMs).
    /// </summary>
    private bool NativeGemmF16 => _opt.Has(DeviceFeatures.NativeGemmF16) && _compiler.CompilesCuda(_opt.Target);

    /// <summary>The tile configuration of the hand-written f16 GEMM with the fewest estimated waves of work for a batch.</summary>
    private int NativeF16Config(long rows, long n, bool dual)
    {
        int sms = Math.Max(1, _opt.ComputeUnits);
        double best = double.MaxValue;
        int config = 0;
        for (int c = 0; c < s_nativeF16TileColumns.Length; c++)
        {
            long columns = dual ? s_nativeF16TileColumns[c] / 2 : s_nativeF16TileColumns[c];
            long tiles = Grid.Div(rows, 128) * ((n + columns - 1) / columns);
            double cost = (double)((tiles + sms - 1) / sms) * s_nativeF16TileColumns[c] / s_nativeF16Speed[c];
            if (cost < best)
            {
                best = cost;
                config = c;
            }
        }

        return config;
    }

    /// <summary>
    /// Launch variants of the hand-written f16 GEMM (single, or dual with <paramref name="dualEpilogue"/>) for batches of
    /// <paramref name="minRows"/> rows and more: one per tile configuration, each serving the row counts it is estimated fastest for.
    /// </summary>
    private KernelVariant[] NativeF16Variants(long n, long k, long maxRows, string? dualEpilogue, Func<Bindings, long> rows, long minRows)
    {
        bool dual = dualEpilogue is not null;
        var used = new SortedSet<int>();
        for (long r = minRows; r <= Math.Max(minRows, maxRows); r += 128)
        {
            used.Add(NativeF16Config(r, n, dual));
        }

        used.Add(NativeF16Config(Math.Max(minRows, maxRows), n, dual));
        return [.. used.Select(c =>
        {
            var macros = Macros(("K", k), ("N", n), ("Y_STRIDE", n), ("DUAL", dual ? 1 : 0), ("H_CFG", c));
            var kernel = _compiler.CompileCuda("gemm_f16_native.cu", dualEpilogue?.Replace("F8N_EPI2", "H_EPI2", StringComparison.Ordinal) ?? "",
                "gemm_f16_native", s_nativeF16Threads[c], s_nativeF16Shared[c], _opt.Target, macros, _opt.Cache);
            long columns = dual ? s_nativeF16TileColumns[c] / 2 : s_nativeF16TileColumns[c];
            return new KernelVariant(b => rows(b) >= minRows && NativeF16Config(rows(b), n, dual) == c, kernel,
                b => new Grid(Grid.Div(rows(b), 128), Grid.Div(n, columns)));
        })];
    }

    /// <summary>
    /// The CUDA C++ form of the dual epilogue <c>epi2</c> (see <see cref="DualEpilogueSource"/>) for the hand-written dual GEMM, as the
    /// <c>F8N_EPI2</c> macro over a device function, or null when the group has an operation without a CUDA form here (the generated
    /// dual GEMM then serves the pair). The expressions are those of the Slang epilogue with single-precision literals and <c>__expf</c>.
    /// </summary>
    private static string? DualEpilogueCudaSource(Node a, Node bNode, List<Node> group, Value output)
    {
        string[] supported = ["Neg", "Exp", "Log", "Sqrt", "Rsqrt", "Sigmoid", "Silu", "Gelu", "GeluTanh", "Tanh", "Relu", "Square", "Recip",
            "Add", "Sub", "Mul", "Max", "Min", "Scale", "Cast", "Clip"];
        if (group.Any(node => !supported.Contains(node.Op.Name)))
        {
            return null;
        }

        var name = new Dictionary<Value, string> { [a.Output] = "m0", [bNode.Output] = "m1" };
        var src = new StringBuilder();
        src.AppendLine("static __device__ __forceinline__ float gelu_tanh(float x) { return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))); }");
        src.AppendLine("static __device__ __forceinline__ float gelu_erf(float x) { return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f)); }");
        src.AppendLine("static __device__ __forceinline__ float sl_epi2(float m0, float m1)");
        src.AppendLine("{");
        foreach (var node in group)
        {
            var local = "v" + name.Count;
            if (!name.ContainsKey(node.Inputs[0]) || node.Inputs.Skip(1).Any(i => !name.ContainsKey(i)))
            {
                // An input from outside the pair (a bias, a mask): not expressible in this epilogue.
                return null;
            }

            src.AppendLine(CultureInfo.InvariantCulture, $"    float {local} = {SinglePrecisionLiterals(OpExpr(node, [.. node.Inputs.Select(i => name[i])]))};");
            name[node.Output] = local;
        }

        src.AppendLine(CultureInfo.InvariantCulture, $"    return {name[output]};");
        src.AppendLine("}");
        src.AppendLine("#define F8N_EPI2(a, b) sl_epi2(a, b)");
        return src.ToString();
    }

    /// <summary>Floating-point literals of a generated expression as <c>float</c> literals (C++ reads <c>1.0</c> as a double).</summary>
    /// <remarks><c>exp</c> becomes <c>__expf</c>, as in the CUDA form of the Slang kernels (<c>common.slang</c>), so the fused epilogue
    /// rounds like the elementwise launch it replaces.</remarks>
    private static string SinglePrecisionLiterals(string expression)
        => ExpCall().Replace(FloatLiteral().Replace(expression, m => m.Value + "f"), "__expf(");

    [GeneratedRegex(@"(?<![\w])exp\(")]
    private static partial Regex ExpCall();

    [GeneratedRegex(@"(?<![\w.])(\d+\.\d*([eE][-+]?\d+)?|\d+[eE][-+]?\d+)(?![\w.])")]
    private static partial Regex FloatLiteral();
}
