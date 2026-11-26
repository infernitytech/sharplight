using System.Buffers.Binary;
using System.Security.Cryptography;
using Sharplight.Models;
using Sharplight.Tensors;
using Xunit;

namespace Sharplight.Import.HuggingFace.Tests;

/// <summary>
/// The weight layouts of the bring-up round 2 releases on tiny synthetic checkpoints: MXFP8 (MiniMax-M3.1: E4M3 codes with U8 E8M0 exponents
/// per 32 values), block FP8 scales tiled per slice of a pre-sharded fused projection (MiMo-V2.6-Pro's qkv_proj), ggml blocks packed in
/// safetensors (ModelOpt MIXED_PRECISION, Nemotron-3.5-Bolt), and the Engram constants DeepSeek-V4.1's reference code derives at load time.
/// </summary>
public sealed class ReleaseQuantLayoutsTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "sharplight-release-quant-" + Guid.NewGuid().ToString("N"));

    public ReleaseQuantLayoutsTests() => Directory.CreateDirectory(_directory);

    public void Dispose() => Directory.Delete(_directory, recursive: true);

    [Fact]
    public void Mxfp8WithExponentByteScalesDecodesToCodeTimesPowerOfTwo()
    {
        var rng = new Random(3);
        const int Rows = 8, Columns = 64;
        byte[] codes = [.. Enumerable.Range(0, Rows * Columns).Select(_ => (byte)(rng.Next(0, 0x7F) | (rng.Next(2) << 7)))];
        byte[] scales = [.. Enumerable.Range(0, Rows * Columns / 32).Select(_ => (byte)rng.Next(110, 140))];
        new SafetensorsWriter()
            .Add("model.layers.0.self_attn.q_proj.weight", "F8_E4M3", [Rows, Columns], codes)
            .Add("model.layers.0.self_attn.q_proj.weight_scale_inv", "U8", [Rows, Columns / 32], scales)
            .Write(Path.Combine(_directory, "model.safetensors"));
        File.WriteAllText(Path.Combine(_directory, "config.json"),
            """{"model_type": "llama", "quantization_config": {"quant_method": "mxfp8", "weight_block_size": [1, 32]}}""");

        using var config = HfConfig.Load(_directory);
        using var model = SafetensorsCheckpoint.Open(_directory);
        var weight = HfQuantizedWeights.Resolve(config, model.Tensors)["model.layers.0.self_attn.q_proj.weight"].ToWeightInfo();
        Assert.Equal(DataType.F8E4M3_B32, weight.Type);
        float[] values = HfTransforms.DecodeF32(weight);
        for (int i = 0; i < Rows * Columns; i++)
        {
            Assert.Equal(HfFp8Codec.E4m3(codes[i]) * MathF.ScaleB(1f, scales[i / 32] - 127), values[i]);
        }

        Assert.Equal(SupportLevel.Supported, HfSupport.GetQuantizationSupport(File.ReadAllText(Path.Combine(_directory, "config.json"))).Level);
    }

    [Fact]
    public void BlockScalesTiledPerSliceOfAFusedProjection()
    {
        // MiMo-V2.6-Pro: 27136 rows stored as 8 slices of 3392 rows, each quantized on its own: ceil(3392 / 128) = 27 scale rows per slice.
        var layout = HfFp8ScaleLayout.Infer(1, 27136, 6144, [216, 48], (128, 128));
        Assert.NotNull(layout);
        Assert.Equal(3392, layout.Value.ChunkRows);
        float[] scales = [.. Enumerable.Range(0, 216 * 48).Select(i => (float)i)];
        Assert.Equal(26 * 48, layout.Value.At(scales, 0, 3391, 0));
        Assert.Equal(27 * 48 + 1, layout.Value.At(scales, 0, 3392, 128));
        Assert.Equal(53 * 48 + 47, layout.Value.At(scales, 0, 2 * 3392 - 1, 6143));
        Assert.Equal(1, layout.Value.At(scales, 0, 127, 128));
    }

    [Fact]
    public void GgmlBlocksPackedInSafetensorsBecomeWeightsOfTheirBlockType()
    {
        var rng = new Random(5);
        byte[] q8 = new byte[4 * 2 * 34];
        rng.NextBytes(q8);
        byte[] iq2 = new byte[2 * 1 * 66];
        rng.NextBytes(iq2);
        new SafetensorsWriter()
            .Add("backbone.layers.0.mixer.q_proj.weight", "U8", [4, 2, 34], q8)
            .Add("backbone.layers.0.mixer.experts.0.up_proj.weight", "U8", [2, 1, 66], iq2)
            .Write(Path.Combine(_directory, "model.safetensors"));
        File.WriteAllText(Path.Combine(_directory, "config.json"), """
            {"model_type": "nemotron_h", "quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "config_groups": {
              "group_0": {"quant_algo": "Q8_0", "group_size": 32, "packing": "ggml", "block_payload_bytes": 34, "targets": ["backbone.layers.0.mixer.q_proj"]},
              "group_2": {"quant_algo": "IQ2_XXS", "group_size": 256, "packing": "ggml", "block_payload_bytes": 66,
                          "targets": ["backbone.layers.0.mixer.experts.0.up_proj"]}}}}
            """);

        using var config = HfConfig.Load(_directory);
        using var model = SafetensorsCheckpoint.Open(_directory);
        var resolved = HfQuantizedWeights.Resolve(config, model.Tensors);
        var q = resolved["backbone.layers.0.mixer.q_proj.weight"].ToWeightInfo();
        Assert.Equal(DataType.Q8_0, q.Type);
        Assert.Equal(new Shape(4, 64), q.Shape);
        Assert.Equal(q8, q.ReadAll());
        var up = resolved["backbone.layers.0.mixer.experts.0.up_proj.weight"].ToWeightInfo();
        Assert.Equal(DataType.IQ2_XXS, up.Type);
        Assert.Equal(new Shape(2, 256), up.Shape);
        Assert.Equal(iq2, up.ReadAll());
    }

    [Fact]
    public void EngramHashConstantsMatchTheReleaseCode()
    {
        // The values llama.cpp's converter wrote into unsloth/DeepSeek-V4.1-Flash-1017-GGUF from the release's engram.py.
        Assert.Equal(
            [76632096046245, 4839876093313, 35959672319349, 73987337458391, 67716810739261, 51510806800915, 30921347202721, 82619226485591],
            DeepseekV41Engram.Multipliers([1, 14], 4, 99092));
        var sizes = DeepseekV41Engram.Primes(2, 4, 8, 16000000);
        Assert.Equal([16000057L, 16000079L, 16000081L], sizes[..3]);
        Assert.Equal(384006168L, sizes[..24].Sum());
        Assert.Equal(384016682L, sizes[24..].Sum());
    }

    [Fact]
    public void EngramTokenMapOfTheReleaseTokenizer()
    {
        string tokenizer = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models", "hf", "DeepSeek-V4.1-Flash-L3-E32", "tokenizer.json");
        Assert.SkipUnless(File.Exists(tokenizer), $"{tokenizer} not present");
        var (map, hashed) = DeepseekV41Engram.TokenMap(tokenizer, 129280);
        Assert.Equal(99092, hashed);
        var bytes = new byte[map.Length * 4];
        for (int i = 0; i < map.Length; i++)
        {
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(i * 4), map[i]);
        }

        // SHA-256 of engram.token_map in unsloth/DeepSeek-V4.1-Flash-1017-GGUF (llama.cpp's converter, the release's build_compressed_token_map).
        Assert.Equal("c60a86322ec17b4142bfef3c57a8d81fb428550cdf487f88a4320cb59fe46481", Convert.ToHexStringLower(SHA256.HashData(bytes)));
    }
}
