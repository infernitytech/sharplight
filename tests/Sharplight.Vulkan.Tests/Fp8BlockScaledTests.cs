using System.Diagnostics;
using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Shaders.Lowering;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

/// <summary>
/// The block-scaled FP8 GEMM (<c>gemm_fp8_tma.cu</c> with <c>F8N_BLOCK_SCALED</c>, <see cref="DeviceFeatures.BlockScaledFp8"/> with <c>SHARPLIGHT_FP8_BLOCK_SCALED=1</c>)
/// against a CPU reference that models its E8M0 rounding: the weights requantized to power-of-two scales on the host
/// (<see cref="Fp8PowerOfTwoScales"/>, itself checked against a rounding reference that enumerates the E4M3 values) and activations
/// that power-of-two scales represent exactly. The GPU run gets the checkpoint's original weights, so its load-time requantization is
/// part of what is checked. The GPU tests run in a child process with the switch set (it is read once per process).
/// </summary>
public sealed class Fp8BlockScaledTests
{
    private const string Switch = "SHARPLIGHT_FP8_BLOCK_SCALED";

    /// <summary>Every finite non-negative E4M3 value, by code.</summary>
    private static readonly double[] s_values = [.. Enumerable.Range(0, 127).Select(c => (double)Fp8PowerOfTwoScales.Value((byte)c))];

    /// <summary>The nearest E4M3 value to <paramref name="v"/>, ties to the even code (the reference of the requantization).</summary>
    private static double Nearest(double v)
    {
        double a = Math.Abs(v), best = 0;
        int code = 0;
        for (int c = 0; c < s_values.Length; c++)
        {
            double d = Math.Abs(s_values[c] - a), e = Math.Abs(best - a);
            if (d < e || (d == e && c % 2 == 0 && code % 2 == 1))
            {
                best = s_values[c];
                code = c;
            }
        }

        return Math.CopySign(best, v);
    }

    /// <summary>
    /// The requantization of one block: the new scale is 2^ceil(log2(amax / 448)) and every new value the nearest E4M3 value (ties to even)
    /// of the old value over it; a second requantization changes no value.
    /// </summary>
    [Fact]
    public void RequantizationRoundsToNearestWithPowerOfTwoScales()
    {
        var rng = new Random(3);
        for (int trial = 0; trial < 400; trial++)
        {
            var codes = new byte[128];
            for (int i = 0; i < 128; i++)
            {
                int c = rng.Next(0, 127) | (rng.Next(2) << 7);
                codes[i] = (byte)(trial % 4 == 0 ? c & 0xBF : c);
            }

            float scale = trial % 7 == 0 ? MathF.ScaleB(1f, rng.Next(-12, 4)) : (float)(rng.NextDouble() * 0.01 + 1e-5);
            var before = codes.Select(c => (double)Fp8PowerOfTwoScales.Value(c) * scale).ToArray();
            float d = scale;
            Fp8PowerOfTwoScales.RequantizeBlock(codes, ref d);
            double amax = before.Max(Math.Abs);
            double expected = amax > 0 ? Math.Pow(2, Math.Ceiling(Math.Log2(amax / 448))) : 1;
            Assert.Equal(expected, d);
            for (int i = 0; i < 128; i++)
            {
                Assert.Equal(Nearest(before[i] / d) * d, (double)Fp8PowerOfTwoScales.Value(codes[i]) * d);
            }

            var after = codes.Select(c => (double)Fp8PowerOfTwoScales.Value(c) * d).ToArray();
            float d2 = d;
            Fp8PowerOfTwoScales.RequantizeBlock(codes, ref d2);
            for (int i = 0; i < 128; i++)
            {
                Assert.Equal(after[i], (double)Fp8PowerOfTwoScales.Value(codes[i]) * d2);
            }
        }
    }

    /// <summary>Raw f8e4m3_b128 rows ({ f32 d; u8 q[128] } blocks) requantized on the host: the reference's weights.</summary>
    private static byte[] Requantized(byte[] raw)
    {
        var copy = (byte[])raw.Clone();
        for (int at = 0; at < copy.Length; at += 132)
        {
            float d = BitConverter.ToSingle(copy, at);
            Fp8PowerOfTwoScales.RequantizeBlock(copy.AsSpan(at + 4, 128), ref d);
            BitConverter.TryWriteBytes(copy.AsSpan(at, 4), d);
        }

        return copy;
    }

    /// <summary>True when this process runs the block-scaled GEMM (child runs on a device with the feature).</summary>
    private static bool Active()
    {
        Gpu.Require();
        return Environment.GetEnvironmentVariable(Switch) == "1" && Gpu.Device!.EnabledFeatures.Contains(DeviceFeatures.BlockScaledFp8);
    }

    private static void Compare(string op, Func<bool, (TG Graph, Value X)> build, int t, int k, int seed, string what)
    {
        var x = Rnd.ExactFp8((long)t * k, seed);
        var gpu = build(false);
        var cpu = build(true);
        var got = Diff.RunVulkan(gpu.Graph.B.Build(), [new Step(t, 0, (gpu.X, x))], maxTokens: t)[0];
        var want = Diff.RunCpu(cpu.Graph.B.Build(), [new Step(t, 0, (cpu.X, x))], maxTokens: t)[0];
        Assert.Equal(want.Length, got.Length);
        for (int i = 0; i < got.Length; i++)
        {
            double err = Math.Abs(got[i] - want[i]) - (2e-3 + 1e-3 * Math.Abs(want[i]));
            Assert.True(err <= 0, $"{op} {what}: [{i}] {got[i]} vs {want[i]}");
        }
    }

    /// <summary>Single matmuls in the 128-row tiles (and the 64-row tiles of the same plan) against the E8M0 reference.</summary>
    [Theory]
    [InlineData(256, 256, 1024)]
    [InlineData(640, 384, 2048)]
    [InlineData(1000, 512, 512)]
    public void MatMulMatchesE8m0Reference(int t, int n, int k)
    {
        if (!Active())
        {
            return;
        }

        var raw = Rnd.WeightBytes(DataType.F8E4M3_B128, (long)n * k, 21, 0.05f);
        (TG, Value) Build(bool reference)
        {
            var tg = new TG();
            var x = tg.InF("x", TG.T, k);
            var w = tg.W("w", DataType.F8E4M3_B128, new Shape(n, k), reference ? Requantized(raw) : raw);
            tg.B.Output("y", tg.B.MatMul(x, w));
            return (tg, x);
        }

        Compare("MatMul.fp8.bs", Build, t, k, 5, $"T={t} N={n} K={k}");
    }

    /// <summary>The dual gate / up pair with SiLU · mul in the store, against the E8M0 reference.</summary>
    [Theory]
    [InlineData(300)]
    [InlineData(1024)]
    public void DualMatchesE8m0Reference(int t)
    {
        if (!Active())
        {
            return;
        }

        const int K = 1024, N = 384;
        var rg = Rnd.WeightBytes(DataType.F8E4M3_B128, (long)N * K, 31, 0.05f);
        var ru = Rnd.WeightBytes(DataType.F8E4M3_B128, (long)N * K, 32, 0.05f);
        (TG, Value) Build(bool reference)
        {
            var tg = new TG();
            var x = tg.InF("x", TG.T, K);
            var g = tg.B.MatMul(x, tg.W("wg", DataType.F8E4M3_B128, new Shape(N, K), reference ? Requantized(rg) : rg));
            var u = tg.B.MatMul(x, tg.W("wu", DataType.F8E4M3_B128, new Shape(N, K), reference ? Requantized(ru) : ru));
            tg.B.Output("y", tg.B.Mul(tg.B.Silu(g), u));
            return (tg, x);
        }

        Compare("MatMul2.fp8.bs", Build, t, K, 11, $"T={t} N={N} K={K}");
    }

    /// <summary>
    /// Child run of <see cref="E4m3StoreMatchesSeparateQuantization"/>: a gate / up pair (SiLU · mul) feeding a down projection, its output
    /// written to the file <c>SL_TEST_FP8_OUT</c> names; checked against the E8M0 reference to the E4M3 rounding of the down input.
    /// </summary>
    [Theory]
    [InlineData(300)]
    [InlineData(1024)]
    public void MlpOutput(int t)
    {
        Gpu.Require();
        if (Environment.GetEnvironmentVariable("SL_TEST_FP8_OUT") is not { } path)
        {
            return;
        }

        // The reference models the E8M0 rounding of the weights only where the block-scaled mode runs.
        bool blockScaled = Active();

        const int K = 1024, N = 512;
        var rg = Rnd.WeightBytes(DataType.F8E4M3_B128, (long)N * K, 41, 0.05f);
        var ru = Rnd.WeightBytes(DataType.F8E4M3_B128, (long)N * K, 42, 0.05f);
        var rd = Rnd.WeightBytes(DataType.F8E4M3_B128, (long)K * N, 43, 0.05f);
        (TG, Value) Build(bool reference)
        {
            var tg = new TG();
            var x = tg.InF("x", TG.T, K);
            byte[] W(byte[] raw) => reference && blockScaled ? Requantized(raw) : raw;
            var g = tg.B.MatMul(x, tg.W("wg", DataType.F8E4M3_B128, new Shape(N, K), W(rg)));
            var u = tg.B.MatMul(x, tg.W("wu", DataType.F8E4M3_B128, new Shape(N, K), W(ru)));
            var h = tg.B.Mul(tg.B.Silu(g), u);
            tg.B.Output("y", tg.B.MatMul(h, tg.W("wd", DataType.F8E4M3_B128, new Shape(K, N), W(rd))));
            return (tg, x);
        }

        var x = Rnd.ExactFp8((long)t * K, 13);
        var gpu = Build(false);
        var cpu = Build(true);
        var got = Diff.RunVulkan(gpu.Item1.B.Build(), [new Step(t, 0, (gpu.Item2, x))], maxTokens: t)[0];
        var want = Diff.RunCpu(cpu.Item1.B.Build(), [new Step(t, 0, (cpu.Item2, x))], maxTokens: t)[0];
        double scale = want.Max(v => Math.Abs((double)v));
        for (int i = 0; i < got.Length; i++)
        {
            // The down input is E4M3-rounded on the GPU only (up to 2^-4 of each value).
            Assert.True(Math.Abs(got[i] - want[i]) <= 0.1 * scale, $"MLP T={t}: [{i}] {got[i]} vs {want[i]} (largest output {scale})");
        }

        File.WriteAllBytes($"{path}.{t}", MemoryMarshal.AsBytes(got.AsSpan()).ToArray());
    }

    /// <summary>
    /// The dual GEMM's E4M3 store (<c>QOUT</c>, from <c>SHARPLIGHT_FP8_QSTORE_MIN_ROWS</c> rows) writes the codes and scales a separate
    /// <c>quantize_fp8</c> writes from its f32 rows: in the block-scaled mode the MLP's outputs are bit-identical with the store on (from
    /// 256 rows) and off.
    /// </summary>
    [Fact]
    public void E4m3StoreMatchesSeparateQuantization()
    {
        Gpu.Require();
        Assert.SkipUnless(Gpu.Device!.EnabledFeatures.Contains(DeviceFeatures.BlockScaledFp8), "The device has no block_scaled_fp8.");
        Assert.SkipWhen(Environment.GetEnvironmentVariable("SL_TEST_FP8_OUT") is not null, "This is the child run.");
        string dir = Path.Combine(Path.GetTempPath(), $"sl-fp8-store-{Environment.ProcessId}");
        Directory.CreateDirectory(dir);
        try
        {
            foreach (var mode in new[] { "1" })
            {
                RunChild([nameof(MlpOutput)], new() { ["SL_TEST_FP8_OUT"] = Path.Combine(dir, "store" + mode), [Switch] = mode }, 2);
                RunChild([nameof(MlpOutput)], new() { ["SL_TEST_FP8_OUT"] = Path.Combine(dir, "separate" + mode), [Switch] = mode, ["SHARPLIGHT_FP8_QSTORE_MIN_ROWS"] = "0" }, 2);
                foreach (int t in new[] { 300, 1024 })
                {
                    Assert.Equal(File.ReadAllBytes(Path.Combine(dir, $"separate{mode}.{t}")), File.ReadAllBytes(Path.Combine(dir, $"store{mode}.{t}")));
                }
            }
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>Runs <paramref name="methods"/> of this class in a child process with <c>SHARPLIGHT_FP8_BLOCK_SCALED=1</c> and <paramref name="environment"/>.</summary>
    private static void RunChild(string[] methods, Dictionary<string, string> environment, int total)
    {
        string host = Environment.ProcessPath ?? throw new InvalidOperationException("no process path");
        var info = new ProcessStartInfo(host) { RedirectStandardOutput = true, RedirectStandardError = true };
        if (Path.GetFileNameWithoutExtension(host) == "dotnet")
        {
            info.ArgumentList.Add(typeof(Fp8BlockScaledTests).Assembly.Location);
        }

        foreach (var method in methods)
        {
            info.ArgumentList.Add("-method");
            info.ArgumentList.Add($"{typeof(Fp8BlockScaledTests).FullName}.{method}");
        }

        info.Environment[Switch] = "1";
        // The child runs are told apart from the outer one by these variables (and may set the switch to 0).
        foreach (var (name, value) in environment)
        {
            info.Environment[name] = value;
        }

        using var process = Process.Start(info) ?? throw new InvalidOperationException("child test run did not start");
        var stderr = process.StandardError.ReadToEndAsync(TestContext.Current.CancellationToken);
        string output = process.StandardOutput.ReadToEnd();
        process.WaitForExit();
        Assert.True(process.ExitCode == 0, $"child run failed ({process.ExitCode}):\n{output}\n{stderr.Result}");
        Assert.Contains($"Total: {total},", output);
        Assert.Contains("Failed: 0,", output);
    }

    /// <summary>Runs the GPU tests of this class in a child process with <c>SHARPLIGHT_FP8_BLOCK_SCALED=1</c>.</summary>
    [Fact]
    public void BlockScaledChildRun()
    {
        Gpu.Require();
        Assert.SkipUnless(Gpu.Device!.EnabledFeatures.Contains(DeviceFeatures.BlockScaledFp8), "The device has no block_scaled_fp8.");
        Assert.SkipWhen(Environment.GetEnvironmentVariable(Switch) == "1", "This is the child run.");
        RunChild([nameof(MatMulMatchesE8m0Reference), nameof(DualMatchesE8m0Reference)], [], 5);
    }
}
