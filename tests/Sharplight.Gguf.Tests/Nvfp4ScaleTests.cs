using System.Buffers.Binary;
using Sharplight.Tensors;
using Xunit;

namespace Sharplight.Gguf.Tests;

/// <summary>
/// NVFP4 tensors with an output multiplier <c>X.scale</c> (LFM3 NVFP4, the Nemotron-3.5 NVFP4 files; llama.cpp multiplies the matmul
/// output by it): per tensor and per expert, each becomes an NVFP4_GS weight whose 40-byte blocks are the multiplier of their matrix
/// followed by the unchanged 36-byte ggml block, read whole and from a range inside blocks. The multiplier and <c>X.input_scale</c>
/// are consumed; NVFP4 tensors without a multiplier stay NVFP4.
/// </summary>
public sealed class Nvfp4ScaleTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "sharplight-nvfp4s-" + Guid.NewGuid().ToString("N"));

    public Nvfp4ScaleTests() => Directory.CreateDirectory(_dir);

    public void Dispose() => Directory.Delete(_dir, recursive: true);

    [Fact]
    public void ScaledNvfp4ReadsAsNvfp4Gs()
    {
        const long Columns = 128, Rows = 3, Experts = 2;
        var rng = new Random(5);
        var experts = new byte[Experts * Rows * Columns / 64 * 36];
        var dense = new byte[Rows * Columns / 64 * 36];
        var plain = new byte[Rows * Columns / 64 * 36];
        rng.NextBytes(experts);
        rng.NextBytes(dense);
        rng.NextBytes(plain);
        float[] expertScales = [1.5e-4f, 3.25e-4f];
        string path = Path.Combine(_dir, "nvfp4.gguf");
        new GgufWriter()
            .AddMetadata("general.architecture", "toy")
            .AddTensorNe("blk.0.ffn_up_exps.weight", DataType.NVFP4, experts, Columns, Rows, Experts)
            .AddTensorNe("blk.0.ffn_up_exps.scale", DataType.F32, Floats(expertScales), Experts)
            .AddTensorNe("blk.0.ffn_down.weight", DataType.NVFP4, dense, Columns, Rows)
            .AddTensorNe("blk.0.ffn_down.scale", DataType.F32, Floats([2.5e-4f]), 1)
            .AddTensorNe("blk.0.ffn_down.input_scale", DataType.F32, Floats([0.5f]), 1)
            .AddTensorNe("blk.0.attn_q.weight", DataType.NVFP4, plain, Columns, Rows)
            .Write(path);

        using var file = GgufFile.Open(path);
        var (converted, consumed) = GgufNvfp4Scales.Find(file.Tensors);
        Assert.Equal(["blk.0.ffn_down.input_scale", "blk.0.ffn_down.scale", "blk.0.ffn_up_exps.scale"], consumed.Order(StringComparer.Ordinal));
        Assert.Equal(["blk.0.ffn_down.weight", "blk.0.ffn_up_exps.weight"], converted.Keys.Order(StringComparer.Ordinal));

        var up = converted["blk.0.ffn_up_exps.weight"];
        Assert.Equal(DataType.NVFP4_GS, up.Type);
        Assert.Equal(new Shape(Experts, Rows, Columns), up.Shape);
        Assert.Equal(up.ByteSize, up.Source.ByteLength);
        byte[] expected = WithMultipliers(experts, blocksPerMatrix: Rows * Columns / 64, expertScales);
        var whole = new byte[up.Source.ByteLength];
        up.Source.Read(0, whole);
        Assert.Equal(expected, whole);
        var part = new byte[57];
        up.Source.Read(33, part);
        Assert.Equal(expected.AsSpan(33, 57).ToArray(), part);

        var down = converted["blk.0.ffn_down.weight"];
        var downBytes = new byte[down.Source.ByteLength];
        down.Source.Read(0, downBytes);
        Assert.Equal(WithMultipliers(dense, blocksPerMatrix: Rows * Columns / 64, [2.5e-4f]), downBytes);
    }

    private static byte[] WithMultipliers(byte[] blocks, long blocksPerMatrix, float[] multipliers)
    {
        long count = blocks.Length / 36;
        var output = new byte[count * 40];
        for (long b = 0; b < count; b++)
        {
            BinaryPrimitives.WriteSingleLittleEndian(output.AsSpan((int)(b * 40)), multipliers[multipliers.Length == 1 ? 0 : b / blocksPerMatrix]);
            blocks.AsSpan((int)(b * 36), 36).CopyTo(output.AsSpan((int)(b * 40 + 4)));
        }

        return output;
    }

    private static byte[] Floats(float[] values) => [.. values.SelectMany(BitConverter.GetBytes)];
}
