using System.Text.Json;
using Sharplight.Models;
using Sharplight.Tensors;
using Xunit;

namespace Sharplight.HuggingFace.Tests;

/// <summary>
/// FP8 checkpoints (vLLM style) through the Hugging Face mappings: a tiny synthetic checkpoint with E4M3 weights and separate scales
/// (DeepSeek block <c>weight_scale_inv</c>, ModelOpt / compressed-tensors per-tensor and per-channel <c>weight_scale</c> with an
/// ignored <c>input_scale</c>) is imported next to an F32 checkpoint holding the same dequantized values. Every canonical weight
/// must decode to exactly the same values, including those that went through row gathers, column-group permutations and expert
/// stacking, and the FP8 weights must keep a block-scaled FP8 type.
/// </summary>
public sealed class Fp8ImportTests : IDisposable
{
    private enum ScaleStyle
    {
        /// <summary>DeepSeek / Qwen FP8: BF16 <c>weight_scale_inv</c> [ceil(N/128), ceil(K/128)].</summary>
        Block,

        /// <summary>ModelOpt: F32 scalar <c>weight_scale</c> plus an <c>input_scale</c>.</summary>
        Tensor,

        /// <summary>compressed-tensors channel strategy: F32 <c>weight_scale</c> [N, 1] plus an <c>input_scale</c>.</summary>
        Channel,
    }

    private sealed record Linear(string Name, long Rows, long Columns, ScaleStyle Style);

    private readonly List<string> _directories = [];
    private readonly List<IDisposable> _open = [];
    private readonly Dictionary<string, float[]> _dense = new(StringComparer.Ordinal);

    public void Dispose()
    {
        foreach (var resource in _open)
        {
            resource.Dispose();
        }

        foreach (var directory in _directories)
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void LlamaBlockScales()
    {
        var config = new Dictionary<string, object>
        {
            ["model_type"] = "llama",
            ["quantization_config"] = new Dictionary<string, object>
            {
                ["quant_method"] = "fp8", ["fmt"] = "e4m3", ["activation_scheme"] = "dynamic", ["weight_block_size"] = new[] { 128, 128 },
            },
        };
        Linear[] linears =
        [
            new("model.layers.0.self_attn.q_proj.weight", 256, 256, ScaleStyle.Block),
            new("model.layers.0.self_attn.k_proj.weight", 128, 256, ScaleStyle.Block),
            new("model.layers.0.mlp.gate_proj.weight", 320, 256, ScaleStyle.Block),    // partial row block
            new("model.layers.0.mlp.down_proj.weight", 256, 320, ScaleStyle.Block),    // K = 320: blocks of 32
        ];
        var weights = ImportAndCompare(config, linears);
        Assert.Equal(DataType.F8E4M3_B128, weights["layers.0.attention.q.weight"].Type);
        Assert.Equal(DataType.F8E4M3_B128, weights["layers.0.mlp.gate.weight"].Type);
        Assert.Equal(DataType.F8E4M3_B32, weights["layers.0.mlp.down.weight"].Type);
    }

    [Fact]
    public void LlamaTensorAndChannelScales()
    {
        var config = new Dictionary<string, object> { ["model_type"] = "llama" };
        Linear[] linears =
        [
            new("model.layers.0.self_attn.q_proj.weight", 256, 256, ScaleStyle.Tensor),
            new("model.layers.0.self_attn.o_proj.weight", 256, 256, ScaleStyle.Channel),
            new("model.layers.0.mlp.down_proj.weight", 256, 96, ScaleStyle.Tensor),
        ];
        var weights = ImportAndCompare(config, linears);
        Assert.Equal(DataType.F8E4M3_B128, weights["layers.0.attention.q.weight"].Type);
        Assert.Equal(DataType.F8E4M3_B128, weights["layers.0.attention.o.weight"].Type);
        Assert.Equal(DataType.F8E4M3_B32, weights["layers.0.mlp.down.weight"].Type);
    }

    /// <summary>Per-expert FP8 tensors (each with its own per-tensor scale) stacked into one [E, N, K] weight.</summary>
    [Fact]
    public void MixtralExpertsStack()
    {
        var config = new Dictionary<string, object> { ["model_type"] = "mixtral", ["num_local_experts"] = 2 };
        Linear[] linears =
        [
            .. Enumerable.Range(0, 2).SelectMany(e => new Linear[]
            {
                new($"model.layers.0.block_sparse_moe.experts.{e}.w1.weight", 64, 128, ScaleStyle.Tensor),
                new($"model.layers.0.block_sparse_moe.experts.{e}.w2.weight", 128, 64, ScaleStyle.Tensor),
            }),
        ];
        var weights = ImportAndCompare(config, linears);
        Assert.Contains(weights.Values, w => w.Shape.Rank == 3 && w.Type == DataType.F8E4M3_B128);
        Assert.Contains(weights.Values, w => w.Shape.Rank == 3 && w.Type == DataType.F8E4M3_B32);
    }

    /// <summary>Qwen3.5 linear attention with more value than key heads: V-head row gathers and out_proj column-group permutation on FP8 blocks.</summary>
    [Fact]
    public void Qwen35LinearAttentionReorders()
    {
        var config = new Dictionary<string, object>
        {
            ["model_type"] = "qwen3_5",
            ["linear_num_key_heads"] = 2, ["linear_num_value_heads"] = 4, ["linear_key_head_dim"] = 128, ["linear_value_head_dim"] = 128,
            ["quantization_config"] = new Dictionary<string, object> { ["quant_method"] = "fp8", ["weight_block_size"] = new[] { 128, 128 } },
        };
        Linear[] linears =
        [
            new("model.language_model.layers.0.linear_attn.in_proj_qkv.weight", 2 * 2 * 128 + 4 * 128, 256, ScaleStyle.Block),
            new("model.language_model.layers.0.linear_attn.in_proj_z.weight", 4 * 128, 256, ScaleStyle.Block),
            new("model.language_model.layers.0.linear_attn.out_proj.weight", 256, 4 * 128, ScaleStyle.Block),
        ];
        var weights = ImportAndCompare(config, linears);
        Assert.Equal(DataType.F8E4M3_B128, weights["layers.0.linear_attention.qkv.weight"].Type);
        Assert.Equal(DataType.F8E4M3_B128, weights["layers.0.linear_attention.out.weight"].Type);

        // Against the element-wise V-head reorder (grouped [numK][vPerK] → tiled [vPerK][numK]) of the dequantized values.
        static float[] Reorder(float[] values, long outer, long inner)
            => System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(
                HfTransforms.ReorderVHeads(HfTransforms.EncodeF32(values), 4, outer, 2, 2, inner)).ToArray();
        var qkv = _dense["model.language_model.layers.0.linear_attn.in_proj_qkv.weight"];
        long qkRows = 2 * 2 * 128;
        float[] expectedQkv = [.. qkv[..(int)(qkRows * 256)], .. Reorder(qkv[(int)(qkRows * 256)..], 1, 128 * 256)];
        Assert.Equal(expectedQkv, HfTransforms.DecodeF32(weights["layers.0.linear_attention.qkv.weight"]));
        var output = _dense["model.language_model.layers.0.linear_attn.out_proj.weight"];
        Assert.Equal(Reorder(output, 256, 128), HfTransforms.DecodeF32(weights["layers.0.linear_attention.out.weight"]));
    }

    /// <summary>
    /// Writes the FP8 checkpoint and its F32 reference, imports both and asserts that every canonical weight decodes identically.
    /// Returns the FP8 import's weights.
    /// </summary>
    private IReadOnlyDictionary<string, WeightInfo> ImportAndCompare(Dictionary<string, object> config, Linear[] linears)
    {
        var rng = new Random(17);
        var fp8 = new SafetensorsWriter();
        var reference = new SafetensorsWriter();
        foreach (var linear in linears)
        {
            _dense[linear.Name] = AddLinear(linear, rng, fp8, reference);
        }

        var quantized = Import(config, fp8);
        var expected = Import(config, reference);
        Assert.Equal(expected.Keys.Order(), quantized.Keys.Order());
        foreach (var (name, weight) in quantized)
        {
            Assert.Equal(expected[name].Shape.ToString(), weight.Shape.ToString());
            Assert.True(weight.Type.IsQuantized, $"{name} lost its FP8 type ({weight.Type})");
            var actual = HfTransforms.DecodeF32(weight);
            Assert.True(HfTransforms.DecodeF32(expected[name]).AsSpan().SequenceEqual(actual), $"{name}: FP8 import differs from the dequantized reference");
        }

        return quantized;
    }

    /// <summary>Random finite E4M3 codes with scales exact in their stored type, written as FP8 + scale and as dequantized F32.</summary>
    private static float[] AddLinear(Linear linear, Random rng, SafetensorsWriter fp8, SafetensorsWriter reference)
    {
        long rows = linear.Rows;
        long columns = linear.Columns;
        var codes = new byte[rows * columns];
        for (long i = 0; i < codes.LongLength; i++)
        {
            byte code;
            do
            {
                code = (byte)rng.Next(256);
            }
            while ((code & 0x7F) == 0x7F);
            codes[i] = code;
        }

        // Scales k/8 * 2^-10, exact in BF16 and F32.
        float NextScale() => (8 + rng.Next(8)) / 8f / 1024f;
        string stem = linear.Name[..^".weight".Length];
        Func<long, long, float> scaleOf;
        switch (linear.Style)
        {
            case ScaleStyle.Block:
            {
                long scaleRows = (rows + 127) / 128;
                long scaleColumns = (columns + 127) / 128;
                var scales = Enumerable.Range(0, (int)(scaleRows * scaleColumns)).Select(_ => NextScale()).ToArray();
                fp8.AddBF16(stem + ".weight_scale_inv", [scaleRows, scaleColumns], scales);
                scaleOf = (r, c) => scales[r / 128 * scaleColumns + c / 128];
                break;
            }
            case ScaleStyle.Tensor:
            {
                float scale = NextScale();
                fp8.AddF32(stem + ".weight_scale", [], [scale]);
                fp8.AddF32(stem + ".input_scale", [1], [0.5f]);
                scaleOf = (_, _) => scale;
                break;
            }
            default:
            {
                var scales = Enumerable.Range(0, (int)rows).Select(_ => NextScale()).ToArray();
                fp8.AddF32(stem + ".weight_scale", [rows, 1], scales);
                fp8.AddF32(stem + ".input_scale", [1], [0.5f]);
                scaleOf = (r, _) => scales[r];
                break;
            }
        }

        fp8.Add(linear.Name, "F8_E4M3", [rows, columns], codes);
        var values = new float[rows * columns];
        for (long r = 0; r < rows; r++)
        {
            for (long c = 0; c < columns; c++)
            {
                values[r * columns + c] = HfFp8Codec.E4m3ToFloat(codes[r * columns + c]) * scaleOf(r, c);
            }
        }

        reference.AddF32(linear.Name, [rows, columns], values);
        return values;
    }

    private IReadOnlyDictionary<string, WeightInfo> Import(Dictionary<string, object> config, SafetensorsWriter tensors)
    {
        string directory = Path.Combine(Path.GetTempPath(), "sharplight-fp8-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        _directories.Add(directory);
        File.WriteAllText(Path.Combine(directory, "config.json"), JsonSerializer.Serialize(config));
        tensors.Write(Path.Combine(directory, "model.safetensors"));

        var hfConfig = HfConfig.Load(directory);
        _open.Add(hfConfig);
        var checkpoint = SafetensorsCheckpoint.Open(directory);
        _open.Add(checkpoint);
        var mapping = HfArchitectureMappings.Find(hfConfig.ModelType) ?? throw new InvalidOperationException(hfConfig.ModelType);
        return mapping.MapWeights(hfConfig, HfFp8Weights.Resolve(hfConfig, checkpoint.Tensors), HfImportOptions.Default);
    }
}
