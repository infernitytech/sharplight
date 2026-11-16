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
    public void DualMatchesSeparateBitwise(string fmt, int t, int n)
    {
        Gpu.Require();
        const int K = 1024;
        float[] Run(bool exposeGate)
        {
            var tg = new TG();
            var x = tg.InF("x", TG.T, K);
            var g = tg.B.MatMul(x, tg.RandW("wg", DataType.Get(fmt), 0.05f, n, K));
            var u = tg.B.MatMul(x, tg.RandW("wu", DataType.Get(fmt), 0.05f, n, K));
            tg.B.Output("y", tg.B.Mul(tg.B.Silu(g), u));
            if (exposeGate)
            {
                tg.B.Output("g", g);
            }

            using var m = CompiledModel.Compile(tg.B.Build(), new PlacementOptions
            {
                Devices = [new DeviceSlot { Device = Gpu.Vulkan! }],
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
}
