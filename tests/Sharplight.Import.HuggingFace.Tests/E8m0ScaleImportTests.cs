using Sharplight.Tensors;
using Xunit;

namespace Sharplight.Import.HuggingFace.Tests;

/// <summary>
/// DeepSeek's own checkpoint layout (DeepSeek-V4: <c>quant_method fp8</c>, <c>scale_fmt ue8m0</c>, <c>expert_dtype fp4</c>), as its
/// reference <c>inference/convert.py</c> reads it: E4M3 weights <c>X.weight</c> with F8_E8M0 <c>X.scale</c> per 128 x 128 block, and
/// FP4 experts as I8 <c>X.weight</c> [N, K/2] (value 2i in the low nibble of byte i) with an F8_E8M0 <c>X.scale</c> per 32 values. A
/// tiny synthetic checkpoint must convert to block-scaled FP8 and MXFP4 and decode to exactly code * 2^(e - 127).
/// </summary>
public sealed class E8m0ScaleImportTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "sharplight-e8m0-" + Guid.NewGuid().ToString("N"));

    public void Dispose() => Directory.Delete(_directory, recursive: true);

    [Fact]
    public void Fp8BlocksAndFp4ExpertsWithE8m0Scales()
    {
        Directory.CreateDirectory(_directory);
        var rng = new Random(7);
        const int Rows = 256, Columns = 256, ExpertRows = 64, ExpertColumns = 128;
        byte[] fp8 = [.. Enumerable.Range(0, Rows * Columns).Select(_ => (byte)(rng.Next(0, 0x7F) | (rng.Next(2) << 7)))];
        byte[] fp8Scales = [.. Enumerable.Range(0, 4).Select(_ => (byte)rng.Next(118, 136))];
        byte[] fp4 = new byte[ExpertRows * ExpertColumns / 2];
        rng.NextBytes(fp4);
        byte[] fp4Scales = [.. Enumerable.Range(0, ExpertRows * ExpertColumns / 32).Select(_ => (byte)rng.Next(110, 140))];
        new SafetensorsWriter()
            .Add("layers.0.attn.wq.weight", "F8_E4M3", [Rows, Columns], fp8)
            .Add("layers.0.attn.wq.scale", SafetensorsTensor.E8m0DType, [2, 2], fp8Scales)
            .Add("layers.0.ffn.experts.0.w1.weight", "I8", [ExpertRows, ExpertColumns / 2], fp4)
            .Add("layers.0.ffn.experts.0.w1.scale", SafetensorsTensor.E8m0DType, [ExpertRows, ExpertColumns / 32], fp4Scales)
            .AddF32("layers.0.attn_norm.scale", [4], [1f, 2f, 3f, 4f])
            .Write(Path.Combine(_directory, "model.safetensors"));
        File.WriteAllText(Path.Combine(_directory, "config.json"),
            """{"model_type": "deepseek_v4", "expert_dtype": "fp4", "quantization_config": {"quant_method": "fp8", "fmt": "e4m3", "scale_fmt": "ue8m0", "weight_block_size": [128, 128]}}""");

        using var config = HfConfig.Load(_directory);
        using var model = SafetensorsCheckpoint.Open(_directory);
        var resolved = HfQuantizedWeights.Resolve(config, model.Tensors);
        Assert.Equal(["layers.0.attn.wq.weight", "layers.0.attn_norm.scale", "layers.0.ffn.experts.0.w1.weight"], resolved.Keys.Order(StringComparer.Ordinal));

        var attention = resolved["layers.0.attn.wq.weight"].ToWeightInfo();
        Assert.Equal(DataType.F8E4M3_B128, attention.Type);
        float[] attentionValues = HfTransforms.DecodeF32(attention);
        for (int i = 0; i < Rows * Columns; i++)
        {
            int block = i / Columns / 128 * 2 + i % Columns / 128;
            Assert.Equal(HfFp8Codec.E4m3(fp8[i]) * MathF.ScaleB(1f, fp8Scales[block] - 127), attentionValues[i]);
        }

        var expert = resolved["layers.0.ffn.experts.0.w1.weight"].ToWeightInfo();
        Assert.Equal(DataType.MXFP4, expert.Type);
        Assert.Equal(new Shape(ExpertRows, ExpertColumns), expert.Shape);
        float[] expertValues = DecodeMxfp4(expert.ReadAll(), ExpertRows * ExpertColumns);
        for (int i = 0; i < ExpertRows * ExpertColumns; i++)
        {
            int code = (fp4[i / 2] >> (i % 2 * 4)) & 0xF;
            Assert.Equal(E2m1(code) * MathF.ScaleB(1f, fp4Scales[i / 32] - 127), expertValues[i]);
        }
    }

    /// <summary>ggml MXFP4 blocks { u8 e; u8 qs[16] }: value j in the low nibble of qs[j], value j + 16 in the high one.</summary>
    private static float[] DecodeMxfp4(byte[] blocks, int count)
    {
        var values = new float[count];
        for (int b = 0; b < count / 32; b++)
        {
            float scale = MathF.ScaleB(1f, blocks[b * 17] - 127);
            for (int j = 0; j < 16; j++)
            {
                byte packed = blocks[b * 17 + 1 + j];
                values[b * 32 + j] = E2m1(packed & 0xF) * scale;
                values[b * 32 + j + 16] = E2m1(packed >> 4) * scale;
            }
        }

        return values;
    }

    private static float E2m1(int code)
    {
        float magnitude = (code & 7) switch { 0 => 0f, 1 => 0.5f, 2 => 1f, 3 => 1.5f, 4 => 2f, 5 => 3f, 6 => 4f, _ => 6f };
        return (code & 8) != 0 ? -magnitude : magnitude;
    }
}
