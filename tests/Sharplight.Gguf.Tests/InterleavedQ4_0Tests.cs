using Sharplight.Tensors;
using Xunit;

namespace Sharplight.Gguf.Tests;

/// <summary>
/// The interleaved Q4_0 types of older llama.cpp releases (Q4_0_4_4, Q4_0_4_8, Q4_0_8_8; bartowski still lists them for Llama 3.x):
/// a tensor written in the interleaved layout (llama.cpp's <c>make_block_q4_0x4</c> / <c>x8</c>: R rows per group, block column by
/// block column, the R scales, then the nibble arrays in chunks interleaved row by row, XOR 0x88) reads back as the plain Q4_0 rows,
/// whole and from any byte range, and the header check counts it as Q4_0. Real files: the deinterleaved file gives llama.cpp's greedy
/// output (docs/model-support.md, sweep).
/// </summary>
public sealed class InterleavedQ4_0Tests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "sharplight-q40x-" + Guid.NewGuid().ToString("N"));

    public InterleavedQ4_0Tests() => Directory.CreateDirectory(_dir);

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    [Theory]
    [InlineData(31, 4, 4)]
    [InlineData(32, 4, 8)]
    [InlineData(33, 8, 8)]
    public void ReadsAsPlainQ4_0(int ggmlType, int rows, int chunk)
    {
        const long columns = 96;
        long rowCount = 2 * rows;
        long blocks = columns / 32;
        var plain = new byte[rowCount * blocks * 18];
        new Random(ggmlType).NextBytes(plain);
        byte[] interleaved = Interleave(plain, rowCount, blocks, rows, chunk);
        string path = Path.Combine(_dir, "q40x.gguf");
        new GgufWriter()
            .AddMetadata("general.architecture", "toy")
            .AddTensorNe("w", ggmlType, DataType.Q4_0, interleaved, columns, rowCount)
            .Write(path);

        using var file = GgufFile.Open(path);
        var weight = file.GetTensor("w").ToWeightInfo();
        Assert.Equal(DataType.Q4_0, weight.Type);
        var whole = new byte[plain.Length];
        weight.Source.Read(0, whole);
        Assert.Equal(plain, whole);
        // A range that starts and ends inside blocks.
        var part = new byte[50];
        weight.Source.Read(25, part);
        Assert.Equal(plain.AsSpan(25, 50).ToArray(), part);
        Assert.Equal(DataType.Q4_0, GgufSupport.GetTensorType(ggmlType));
    }

    // llama.cpp's repack: for every group of R rows and block column, the R scales, then the nibble chunks of the R blocks interleaved.
    private static byte[] Interleave(byte[] plain, long rowCount, long blocks, int rows, int chunk)
    {
        var output = new byte[plain.Length];
        long at = 0;
        for (long group = 0; group < rowCount / rows; group++)
        {
            for (long column = 0; column < blocks; column++)
            {
                for (int r = 0; r < rows; r++)
                {
                    long source = ((group * rows + r) * blocks + column) * 18;
                    output[at++] = plain[source];
                    output[at++] = plain[source + 1];
                }

                for (int i = 0; i < rows * 16 / chunk; i++)
                {
                    long source = ((group * rows + i % rows) * blocks + column) * 18 + 2 + i / rows * chunk;
                    for (int b = 0; b < chunk; b++)
                    {
                        output[at++] = (byte)(plain[source + b] ^ 0x88);
                    }
                }
            }
        }

        return output;
    }
}
