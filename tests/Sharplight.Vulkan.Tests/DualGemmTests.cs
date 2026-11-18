using System.Runtime.InteropServices;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

/// <summary>
/// The dual gemm_cm2 (gate / up with f16 or bf16 weights and the SiLU·mul epilogue) serves prefill batches on 128-token tiles
/// where the separate GEMMs would not split K, and must give bit-identical results to them there (greedy output of a prompt must
/// not depend on the fusion). The second graph also outputs the gate product, which keeps its pair on the separate GEMMs.
/// </summary>
public sealed class DualGemmTests
{
    [Theory]
    [InlineData("bf16", 512, 8192)]
    [InlineData("bf16", 636, 4104)]
    [InlineData("f16", 1100, 8192)]
    [InlineData("f16", 100, 96, 64, "Gelu")]
    [InlineData("f16", 77, 128, 64, "GeluTanh")]
    [InlineData("q4_k", 512, 4096, 1024)]
    [InlineData("q6_k", 300, 2048, 1024)]
    [InlineData("q5_k", 640, 1024, 2048)]
    [InlineData("q4_0", 512, 2048, 1024)]
    public void DualMatchesSeparateBitwise(string fmt, int t, int n, int k = 1024, string activation = "Silu")
    {
        Gpu.Require();
        int K = k;
        float[] Run(bool exposeGate)
        {
            var tg = new TG();
            var x = tg.InF("x", TG.T, K);
            var g = tg.B.MatMul(x, tg.RandW("wg", DataType.Get(fmt), 0.05f, n, K));
            var u = tg.B.MatMul(x, tg.RandW("wu", DataType.Get(fmt), 0.05f, n, K));
            tg.B.Output("y", tg.B.Mul(activation == "Silu" ? tg.B.Silu(g) : tg.B.Unary(activation, g), u));
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
            var y = new float[(long)t * n];
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

    /// <summary>
    /// A down projection reading the dual GEMM's output takes the f16 copy the dual GEMM stores next to it (no conversion launch);
    /// its result must equal that of the separate launches, which convert the elementwise output.
    /// </summary>
    [Theory]
    [InlineData("bf16", 512, 4096, 1024)]
    [InlineData("bf16", 600, 4160, 1024)]
    [InlineData("f16", 1100, 2048, 1024)]
    [InlineData("q4_k", 600, 4096, 1024)]
    public void DownProjectionOfDualMatchesSeparateBitwise(string fmt, int t, int n, int k)
    {
        Gpu.Require();
        float[] Run(bool exposeGate)
        {
            var tg = new TG();
            var x = tg.InF("x", TG.T, k);
            var g = tg.B.MatMul(x, tg.RandW("wg", DataType.Get(fmt), 0.05f, n, k));
            var u = tg.B.MatMul(x, tg.RandW("wu", DataType.Get(fmt), 0.05f, n, k));
            tg.B.Output("z", tg.B.MatMul(tg.B.Mul(tg.B.Silu(g), u), tg.RandW("wd", DataType.Get(fmt), 0.05f, k, n)));
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
            m.SetInput(x, MemoryMarshal.AsBytes(Rnd.F((long)t * k, 9).AsSpan()));
            var p = new RunParameters();
            p.Symbols["T"] = t;
            m.Run(p);
            var z = new float[(long)t * k];
            m.GetOutput<float>("z", z);
            return z;
        }

        var dual = Run(exposeGate: false);
        var separate = Run(exposeGate: true);
        for (int i = 0; i < dual.Length; i++)
        {
            Assert.True(BitConverter.SingleToInt32Bits(dual[i]) == BitConverter.SingleToInt32Bits(separate[i]),
                $"T={t}: element {i} differs ({dual[i]:R} vs {separate[i]:R})");
        }
    }
}
