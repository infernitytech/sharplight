using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

public sealed class ProbeMvRows
{
    [Fact]
    public void Probe()
    {
        string? dir = Environment.GetEnvironmentVariable("PROBE_DIR");
        if (dir is null)
        {
            return;
        }

        Gpu.Require();
        foreach (var fmt in new[] { "bf16", "f16", "f32", "q8_0", "q4_k", "q6_k", "q5_k", "q4_0", "iq4_xs", "f8e4m3_b128" })
        {
            foreach (var (k, n) in new[] { (1024, 3584), (1024, 6144), (1536, 8960), (8960, 1536), (2048, 4096), (3072, 4096), (3584, 1024), (4096, 2816), (4096, 4096), (5120, 6144), (6144, 5120), (8192, 2048), (17408, 5120) })
            {
                var tg = new TG();
                var x = tg.InF("x", TG.T, k);
                tg.B.Output("y", tg.B.MatMul(x, tg.RandW("w", DataType.Get(fmt), 0.1f, n, k)));
                var data = Rnd.F(k, 7);
                var outputs = Diff.RunVulkan(tg.B.Build(), [new Step(1, 0, (x, data))], maxTokens: 16);
                var bytes = new byte[outputs[0].Length * 4];
                Buffer.BlockCopy(outputs[0], 0, bytes, 0, bytes.Length);
                File.WriteAllBytes(Path.Combine(dir, $"{fmt}_{k}_{n}.bin"), bytes);
            }
        }
    }
}
