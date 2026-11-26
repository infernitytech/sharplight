using System.Diagnostics;
using System.Runtime.InteropServices;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

/// <summary>
/// The fp8 tensor-core GEMM (<c>gemm_fp8.slang</c>, prefill batches of 65+ rows with block-scaled E4M3 weights, needs
/// <c>matrix_fp8</c>) against the CPU oracle, and the f16 tensor-core GEMM that replaces it when the feature is disabled.
/// </summary>
public sealed class Fp8GemmTests
{
    private static readonly string[] s_formats = ["f8e4m3_b128", "f8e4m3_b32"];

    public static IEnumerable<object[]> Cases()
    {
        foreach (var fmt in s_formats)
        {
            foreach (var (t, n) in new[] { (17, 96), (40, 100), (65, 96), (200, 100), (300, 520) })
            {
                yield return [fmt, t, n];
            }
        }
    }

    public static IEnumerable<object[]> Formats() => s_formats.Select(f => new object[] { f });

    private static void Run(string fmt, int t, int n, int k, float[] x, double atol, double rtol, IReadOnlySet<string>? disabled = null, int maxTokens = 0)
    {
        Gpu.Require();
        var tg = new TG();
        var xv = tg.InF("x", TG.T, k);
        var w = tg.RandW("w", DataType.Get(fmt), 1f, n, k);
        tg.B.Output("y", tg.B.MatMul(xv, w));
        Diff.Check("MatMul.fp8." + fmt, tg.B.Build(), new Step(t, 0, (xv, x)), atol: atol, rtol: rtol, maxTokens: Math.Max(maxTokens, Math.Max(t, 64)),
            what: $"T={t} N={n} K={k}", disabledFeatures: disabled);
    }

    /// <summary>
    /// Absolute tolerance of the cases whose activations E4M3 represents exactly. Every product is then exact, so the only error left
    /// is the summation, and that depends on the tensor cores: Blackwell's f8 x f8 -> f32 MMAs accumulate in full f32 (max |d| about
    /// 1e-5 here), while Ada's (RTX 4090; Hopper too, as documented for DeepSeek-V3's FP8 training) align the products of an MMA to the
    /// largest one and keep only about 14 fraction bits. A CPU model of that accumulator (products and partial sum truncated to 13-14
    /// bits below the largest exponent, per 32-value MMA, chained over a 128-value weight block) reproduces the RTX 4090 results: max
    /// |d| 3e-3 to 4e-3 at 14 bits and 9e-3 at 13 bits, for outputs of RMS 5-15, whatever K. A structural defect (a K step lost or
    /// summed twice in a stream-K or split-K tile, a wrong block scale, unwritten rows of a partial tile) errs by about the RMS of a
    /// block's contribution, 0.5 or more, so this tolerance still sits far below what the tests are for.
    /// </summary>
    private const double ExactAtol = 1.5e-2;

    /// <summary>
    /// Relative RMS error bound of the exact-activation cases: 1.3e-4 for the 13-bit accumulator model above, about 1.6e-6 on
    /// Blackwell. It keeps the check of the whole output tight where the per-element tolerance has to allow for the tails.
    /// </summary>
    private const double ExactMaxRelativeRms = 3e-4;

    /// <summary>An fp8 GEMM with activations E4M3 represents exactly (<see cref="Rnd.ExactFp8"/>) against the f32 CPU oracle.</summary>
    private static void RunExact(string fmt, int t, int n, int k, int seed, int maxTokens = 0)
    {
        Run(fmt, t, n, k, Rnd.ExactFp8((long)t * k, seed), atol: ExactAtol, rtol: 1e-3, maxTokens: maxTokens);
        double rel = Diff.RelativeRms[$"MatMul.fp8.{fmt} T={t} N={n} K={k}"];
        Assert.True(rel <= ExactMaxRelativeRms, $"MatMul.fp8.{fmt} T={t} N={n} K={k}: relative rms {rel:E2} > {ExactMaxRelativeRms:E2}");
    }

    /// <summary>Random activations: E4M3 keeps 4 significant bits, so each product errs by up to ~3 %.</summary>
    [Theory]
    [MemberData(nameof(Cases))]
    public void RandomActivations(string fmt, int t, int n)
    {
        const int K = 512;
        Run(fmt, t, n, K, Rnd.F((long)t * K, 3), atol: 1.5, rtol: 5e-2);
    }

    /// <summary>Activations that per-128 E4M3 quantization represents exactly: the fp8 GEMM must then match the f32 reference (up to the tensor cores' summation, see <see cref="ExactAtol"/>).</summary>
    [Theory]
    [MemberData(nameof(Cases))]
    public void ExactActivations(string fmt, int t, int n)
    {
        const int K = 1024;
        RunExact(fmt, t, n, K, seed: 5);
    }

    /// <summary>
    /// Batches a few rows past a 128-row token tile, which the CUDA provider's tensor-copy GEMM runs in its partial form (warps with
    /// 1-16, 17-32 or no rows of the batch in the last tile: 266, 280, 300 and 344 rows), against the CPU oracle with activations that
    /// E4M3 represents exactly.
    /// </summary>
    [Theory]
    [InlineData(266)]
    [InlineData(280)]
    [InlineData(300)]
    [InlineData(344)]
    public void PartialTokenTiles(int t)
    {
        const int K = 1024, N = 6144;
        RunExact("f8e4m3_b128", t, N, K, seed: 13, maxTokens: 512);
    }

    /// <summary>
    /// Batches whose 128-row tiles do not fill whole waves of the SMs, which the CUDA provider's tensor-copy GEMM runs stream-K (the K
    /// steps of the last waves' tiles shared evenly by every SM's block; a tile split between blocks sums their partial sums in the
    /// block holding its first K step), against the CPU oracle with activations that E4M3 represents exactly: 128 rows (32 tiles, each
    /// split across two or three blocks), 256 and 266 rows (with the partial form), 130 rows of 10,752 outputs (170 tiles: two full waves walked tile by tile, then
    /// the rest shared).
    /// </summary>
    [Theory]
    [InlineData(200, 2048, 8192)]
    [InlineData(256, 4096, 8192)]
    [InlineData(266, 4096, 8192)]
    [InlineData(130, 10752, 1024)]
    public void StreamKTiles(int t, int n, int k)
    {
        RunExact("f8e4m3_b128", t, n, k, seed: 17, maxTokens: Math.Max(t, 512));
    }

    /// <summary>
    /// A batch whose shared tiles fill at least a wave (266 rows of 4,096 outputs: 96 tiles) runs the CUDA provider's tensor-copy GEMM
    /// stream-K with each cut tile continued from the previous block's accumulators: its outputs equal bit for bit those of the same rows
    /// in a batch of 2,100 rows, whose 17 token tiles walk tile by tile.
    /// </summary>
    [Fact]
    public void StreamKMatchesTileWalkBitwise()
    {
        Gpu.RequireCuda();
        var (part, whole) = RunPrefix("f8e4m3_b128", 266, 2100, 4096, 8192);
        Assert.Equal(whole, part);
    }

    /// <summary>
    /// The outputs of the first <paramref name="rows"/> rows of a [<paramref name="n"/>, <paramref name="k"/>] matmul of <paramref name="fmt"/>
    /// weights run as a batch of <paramref name="rows"/> and as one of <paramref name="wholeRows"/> rows (bit patterns).
    /// </summary>
    private static (int[] Part, int[] Whole) RunPrefix(string fmt, int rows, int wholeRows, int n, int k)
    {
        var tg = new TG();
        var x = tg.InF("x", TG.T, k);
        tg.B.Output("y", tg.B.MatMul(x, tg.RandW("w", DataType.Get(fmt), 0.05f, n, k)));
        using var m = CompiledModel.Compile(tg.B.Build(), new PlacementOptions
        {
            Devices = [new DeviceSlot { Device = Gpu.Device! }],
            Compile = new CompileOptions { MaxTokens = wholeRows },
            ReservePerDevice = 0,
        });
        var input = Rnd.F((long)wholeRows * k, 21);
        int[] Run(int t)
        {
            m.SetInput(x, MemoryMarshal.AsBytes(input.AsSpan(0, t * k)));
            var p = new RunParameters();
            p.Symbols["T"] = t;
            m.Run(p);
            var y = new float[(long)t * n];
            m.GetOutput<float>("y", y);
            return [.. y.Take(rows * n).Select(BitConverter.SingleToInt32Bits)];
        }

        return (Run(rows), Run(wholeRows));
    }

    /// <summary>
    /// Row counts below the compiled maximum, in the ranges where the fp8 GEMM with <c>SHARPLIGHT_FP8_SHORT_TILES=1</c> takes 64-row
    /// tiles above 64 rows (129-192, 257-320) and splits K: every (tile height, split) of the row counts a step may bind has its
    /// launch variant (the split variants were probed every 128 rows, so those buckets had none and stored nothing).
    /// </summary>
    [Theory]
    [InlineData("f8e4m3_b128", 150)]
    [InlineData("f8e4m3_b128", 182)]
    [InlineData("f8e4m3_b128", 300)]
    [InlineData("f8e4m3_b32", 171)]
    public void ShortTilesBelowMaxTokens(string fmt, int t)
    {
        const int K = 4096;
        RunExact(fmt, t, 4096, K, seed: 9, maxTokens: 2048);
    }

    /// <summary><see cref="ShortTilesBelowMaxTokens"/> with the opt-in 64-row tiles (read once per process, so in a child run).</summary>
    [Fact]
    public void ShortTilesBelowMaxTokensOptIn()
    {
        Gpu.Require();
        string method = $"{typeof(Fp8GemmTests).FullName}.{nameof(ShortTilesBelowMaxTokens)}";
        string host = Environment.ProcessPath ?? throw new InvalidOperationException("no process path");
        var info = new ProcessStartInfo(host) { RedirectStandardOutput = true, RedirectStandardError = true };
        if (Path.GetFileNameWithoutExtension(host) == "dotnet")
        {
            info.ArgumentList.Add(typeof(Fp8GemmTests).Assembly.Location);
        }

        info.ArgumentList.Add("-method");
        info.ArgumentList.Add(method);
        info.Environment["SHARPLIGHT_FP8_SHORT_TILES"] = "1";
        using var process = Process.Start(info) ?? throw new InvalidOperationException("child test run did not start");
        var stderr = process.StandardError.ReadToEndAsync(TestContext.Current.CancellationToken);
        string output = process.StandardOutput.ReadToEnd();
        process.WaitForExit();
        Assert.True(process.ExitCode == 0, $"child run failed ({process.ExitCode}):\n{output}\n{stderr.Result}");
        Assert.Contains("Total: 4,", output);
        Assert.Contains("Failed: 0,", output);
    }

    /// <summary>A grid large enough to run without split-K (and the 128-token tile with partial row and column tiles).</summary>
    [Theory]
    [MemberData(nameof(Formats))]
    public void LargeGrid(string fmt)
    {
        const int K = 256;
        const int T = 1000;
        RunExact(fmt, T, 3000, K, seed: 7);
    }

    /// <summary>
    /// The dual fp8 GEMM (gate / up with the SiLU·mul epilogue) serves prefill batches above 64 rows where the separate GEMMs
    /// would not split K, and must give bit-identical results to them there (greedy output of a prompt must not depend on the
    /// fusion). The second graph also outputs the gate product, which keeps its pair on the separate GEMMs.
    /// </summary>
    [Theory]
    [InlineData(266)]
    [InlineData(300)]
    [InlineData(512)]
    [InlineData(636)]
    public void DualMatchesSeparateBitwise(int t)
    {
        Gpu.Require();
        const int K = 1024, N = 8192;
        float[] Run(bool exposeGate)
        {
            var tg = new TG();
            var x = tg.InF("x", TG.T, K);
            var g = tg.B.MatMul(x, tg.RandW("wg", DataType.Get("f8e4m3_b128"), 0.05f, N, K));
            var u = tg.B.MatMul(x, tg.RandW("wu", DataType.Get("f8e4m3_b128"), 0.05f, N, K));
            tg.B.Output("y", tg.B.Mul(tg.B.Silu(g), u));
            if (exposeGate)
            {
                tg.B.Output("g", g);
            }

            using var m = CompiledModel.Compile(tg.B.Build(), new PlacementOptions
            {
                Devices = [new DeviceSlot { Device = Gpu.Device! }],
                Compile = new CompileOptions { MaxTokens = t },
                ReservePerDevice = 0,
            });
            m.SetInput(x, MemoryMarshal.AsBytes(Rnd.F((long)t * K, 9).AsSpan()));
            var p = new RunParameters();
            p.Symbols["T"] = t;
            m.Run(p);
            var y = new float[(long)t * N];
            m.GetOutput<float>("y", y);
            return y;
        }

        var dual = Run(exposeGate: false);
        var separate = Run(exposeGate: true);
        for (int i = 0; i < dual.Length; i++)
        {
            Assert.True(BitConverter.SingleToInt32Bits(dual[i]) == BitConverter.SingleToInt32Bits(separate[i]),
                $"T={t}: element {i} differs ({dual[i]:R} vs {separate[i]:R})");
        }
    }

    /// <summary>With <c>matrix_fp8</c> disabled the same batches run on the f16 tensor-core GEMM.</summary>
    /// <summary>
    /// Decode-sized batches of weights in the hand-written FP8 GEMM's 128-row groups (N % 128 == 0, K % 512 == 0): on CUDA, 9-16 rows
    /// run the register-streaming <c>gemm_fp8_decode.cu</c> (K split over 1-6 warps by shape), 17-64 rows the decode form of
    /// <c>gemm_fp8_native.cu</c>; with exactly representable activations they must match the f32 reference (to f16 rounding: the
    /// skinny GEMM, which serves these rows on Vulkan, reads an f16 copy of x).
    /// </summary>
    [Theory]
    [InlineData(9, 256, 1536)]
    [InlineData(12, 384, 1024)]
    [InlineData(16, 256, 3072)]
    [InlineData(16, 1024, 512)]
    [InlineData(20, 256, 1536)]
    [InlineData(40, 384, 1024)]
    public void DecodeRowsGroupedWeights(int t, int n, int k)
        => Run("f8e4m3_b128", t, n, k, Rnd.ExactFp8((long)t * k, 7), atol: 1e-2, rtol: 1e-2);

    /// <summary>
    /// The gate / up pair (SiLU · mul) at decode sizes on grouped E4M3 weights: from 9 rows the dual decode form of the hand-written
    /// FP8 GEMM with the epilogue in its store (the skinny GEMM serves 1-8 rows), against the CPU.
    /// </summary>
    [Theory]
    [InlineData(4)]
    [InlineData(9)]
    [InlineData(16)]
    [InlineData(24)]
    public void DualDecodeRowsGroupedWeights(int t)
    {
        Gpu.Require();
        const int K = 1024, N = 384;
        var tg = new TG();
        var x = tg.InF("x", TG.T, K);
        var g = tg.B.MatMul(x, tg.RandW("wg", DataType.Get("f8e4m3_b128"), 0.05f, N, K));
        var u = tg.B.MatMul(x, tg.RandW("wu", DataType.Get("f8e4m3_b128"), 0.05f, N, K));
        tg.B.Output("y", tg.B.Mul(tg.B.Silu(g), u));
        Diff.Check("MatMul2.fp8.decode", tg.B.Build(), new Step(t, 0, (x, Rnd.ExactFp8((long)t * K, 11))), atol: 2e-3, rtol: 1e-2, maxTokens: 64,
            what: $"T={t} N={N} K={K}");
    }

    [Theory]
    [MemberData(nameof(Formats))]
    public void Fp8MatrixDisabled(string fmt)
    {
        const int K = 512;
        const int T = 130;
        Run(fmt, T, 96, K, Rnd.F((long)T * K, 3), atol: 0.25, rtol: 1e-2, disabled: new HashSet<string> { DeviceFeatures.MatrixFp8 });
    }
}
