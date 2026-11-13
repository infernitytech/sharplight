using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

/// <summary>
/// Gathers from quantized tables of several GiB (the hashed n-gram table of Qwen3.8-Flash-Next is 28.8 GB of IQ4_NL), so byte
/// offsets of the selected rows do not fit 32 bits. Opt-in: set SHARPLIGHT_TEST_LARGE=1 (needs that much free host and device memory).
/// </summary>
public class LargeTableGatherTests
{
    [Theory]
    [InlineData(3L << 30)]
    [InlineData(6L << 30)]
    public unsafe void Gather_iq4_nl_rows_beyond_32_bit_offsets(long tableBytes)
    {
        Assert.SkipUnless(Environment.GetEnvironmentVariable("SHARPLIGHT_TEST_LARGE") == "1", "large test: set SHARPLIGHT_TEST_LARGE=1 to run it");
        Gpu.Require();
        const int D = 160;
        long rowBytes = DataType.Get("iq4_nl").ByteSize(D);
        long rows = tableBytes / rowBytes;
        using var source = new NativeSource(rows * rowBytes, (ptr, bytes) => Fill((byte*)ptr, bytes));
        var tg = new TG();
        tg.Weights["tbl"] = new WeightInfo { Name = "tbl", Type = DataType.Get("iq4_nl"), Shape = new Shape(rows, D), Source = source };
        var idx = tg.In("idx", DataType.I32, TG.T);
        tg.B.Output("y", tg.B.Gather(tg.B.Weight("tbl"), idx));
        int[] ids = [0, 1, (int)(rows / 2), (int)(rows - 2), (int)(rows - 1)];
        Diff.Check("Gather.iq4_nl.large", tg.B.Build(), new Step(ids.Length, 0, (idx, ids)), atol: 1e-6, rtol: 1e-6);
    }

    // IQ4_NL blocks (f16 scale + 16 bytes of nibbles) whose scale and nibbles vary with the byte offset.
    private static unsafe void Fill(byte* p, long bytes)
    {
        const long Block = 18;
        const long BlocksPerChunk = 1L << 20;
        long blocks = bytes / Block;
        Parallel.For(0, (blocks + BlocksPerChunk - 1) / BlocksPerChunk, chunk =>
        {
            long end = Math.Min(blocks, (chunk + 1) * BlocksPerChunk);
            for (long block = chunk * BlocksPerChunk; block < end; block++)
            {
                byte* q = p + block * Block;
                ulong h = (ulong)block * 0x9E3779B97F4A7C15UL;
                *(Half*)q = (Half)(0.5f + (h >> 60) / 16f);
                for (int i = 2; i < Block; i++)
                {
                    q[i] = (byte)(h >> (i * 3 % 56));
                }
            }
        });
    }
}
