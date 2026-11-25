using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

/// <summary>
/// E4M3 weights whose 128-row groups share every block's scale (<see cref="WeightInfo.ScaleRowBlock"/> 128, as 128 x 128 scale-block
/// checkpoints import): the prompt-sized FP8 GEMM reads one weight scale per group (<c>gemm_fp8_tma.cu</c> <c>F8N_WUNI</c> on CUDA),
/// with the same results bit for bit as the same weights without the fact; the products of a single matmul, a gate / up pair and the
/// MLP behind it, at token counts on both tile heights.
/// </summary>
public sealed class Fp8UniformScalesTests
{
    private const int K = 1024, N = 512;

    /// <summary>Raw f8e4m3_b128 rows of <paramref name="rows"/> x <paramref name="k"/> with each block's scale copied from row 0 of its 128-row group.</summary>
    private static byte[] Uniform(byte[] raw, long rows, long k)
    {
        long rowBytes = k / 128 * 132;
        for (long r = 0; r < rows; r++)
        {
            long group = r / 128 * 128;
            for (long b = 0; b < k / 128; b++)
            {
                Array.Copy(raw, group * rowBytes + b * 132, raw, r * rowBytes + b * 132, 4);
            }
        }

        return raw;
    }

    private static Value W(TG tg, string name, long rows, long k, byte[] data, bool known)
    {
        var v = tg.W(name, DataType.F8E4M3_B128, new Shape(rows, k), data);
        var w = tg.Weights[name];
        tg.Weights[name] = new WeightInfo { Name = w.Name, Type = w.Type, Shape = w.Shape, Source = w.Source, ScaleRowBlock = known ? 128 : 0 };
        return v;
    }

    [Theory]
    [InlineData(300)]
    [InlineData(1024)]
    public void SameResultsWithAndWithoutTheFact(int t)
    {
        Gpu.Require();
        var rg = Uniform(Rnd.WeightBytes(DataType.F8E4M3_B128, (long)N * K, 51, 0.05f), N, K);
        var ru = Uniform(Rnd.WeightBytes(DataType.F8E4M3_B128, (long)N * K, 52, 0.05f), N, K);
        var rd = Uniform(Rnd.WeightBytes(DataType.F8E4M3_B128, (long)K * N, 53, 0.05f), K, N);
        (TG, Value) Build(bool known)
        {
            var tg = new TG();
            var x = tg.InF("x", TG.T, K);
            var g = tg.B.MatMul(x, W(tg, "wg", N, K, rg, known));
            var u = tg.B.MatMul(x, W(tg, "wu", N, K, ru, known));
            var h = tg.B.Mul(tg.B.Silu(g), u);
            tg.B.Output("single", tg.B.MatMul(x, W(tg, "ws", N, K, rg, known)));
            tg.B.Output("y", tg.B.MatMul(h, W(tg, "wd", K, N, rd, known)));
            return (tg, x);
        }

        var x = Rnd.ExactFp8((long)t * K, 17);
        var with = Build(true);
        var without = Build(false);
        var a = Diff.RunVulkan(with.Item1.B.Build(), [new Step(t, 0, (with.Item2, x))], maxTokens: t);
        var b = Diff.RunVulkan(without.Item1.B.Build(), [new Step(t, 0, (without.Item2, x))], maxTokens: t);
        Assert.Equal(b.Count, a.Count);
        for (int o = 0; o < a.Count; o++)
        {
            Assert.Equal(b[o], a[o]);
        }

        var cpu = Diff.RunCpu(without.Item1.B.Build(), [new Step(t, 0, (without.Item2, x))], maxTokens: t)[0];
        double scale = cpu.Max(v => Math.Abs((double)v));
        for (int i = 0; i < cpu.Length; i++)
        {
            Assert.True(Math.Abs(a[0][i] - cpu[i]) <= 0.1 * scale, $"T={t}: [{i}] {a[0][i]} vs {cpu[i]}");
        }
    }
}
