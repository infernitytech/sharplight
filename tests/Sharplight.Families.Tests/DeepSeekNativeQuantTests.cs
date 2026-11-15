using System.Buffers.Binary;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using Sharplight.Cpu;
using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Vulkan;

namespace Sharplight.Families.Tests;

/// <summary>
/// DeepSeek-V4 in DeepSeek's own quantized layout (the released DeepSeek-V4-Flash: <c>quant_method fp8</c>, <c>scale_fmt ue8m0</c>,
/// <c>expert_dtype fp4</c>). The tiny random DeepSeek-V4 is quantized the way DeepSeek's <c>inference/kernel.py</c> does: attention and
/// shared-expert projections to E4M3 with one power-of-two (F8_E8M0) scale per 128 x 128 block, routed experts to E2M1 with one E8M0
/// scale per 32 values (I8 bytes, value 2i in the low nibble). A second checkpoint holds exactly the dequantized values in F32. Both run
/// the same teacher-forced ids (one prefill chunk, then single tokens) and their logits must agree: on the CPU to summation order, on
/// Vulkan within the bound of the other quantized checkpoints (f16 tensor-core prefill). Both run without the 8-bit integer-dot matvec,
/// as the family cases of this model do: its random indexer has near-tied top-k picks, and the 8-bit activations of that matvec (which
/// only the quantized weights take) move the logits by 16 % of the spread. Skips when the tiny checkpoint is absent.
/// </summary>
public sealed partial class DeepSeekNativeQuantTests : IDisposable
{
    private const int PrefillChunk = 17;
    private const int Positions = 24;

    private static readonly string Home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
    private static readonly float[] s_e2m1 = [0f, 0.5f, 1f, 1.5f, 2f, 3f, 4f, 6f];
    private static readonly float[] s_e4m3 = [.. Enumerable.Range(0, 0x7F).Select(code => E4m3((byte)code))];

    private readonly string _root = Path.Combine(Path.GetTempPath(), "sharplight-dsv4-native-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_root))
        {
            Directory.Delete(_root, recursive: true);
        }
    }

    [GeneratedRegex(@"\.(attn\.(wq_a|wq_b|wkv|wo_a|wo_b)|shared_experts\.w[123])\.weight$")]
    private static partial Regex Fp8Weight();

    [GeneratedRegex(@"\.experts\.\d+\.w[123]\.weight$")]
    private static partial Regex Fp4Weight();

    [Theory]
    [InlineData(false, 1e-4)]
    [InlineData(true, 0.04)]
    public void MatchesItsDequantizationOnTheDevice(bool vulkan, double maxRelative)
    {
        string source = Path.Combine(Home, "models", "hf", "tiny-random-deepseek-v4");
        Assert.SkipUnless(File.Exists(Path.Combine(source, "model.safetensors")), $"{source} not present");
        var (quantized, dequantized, fp8Count, fp4Count) = Build(source);
        Assert.True(fp8Count > 0 && fp4Count > 0, $"{fp8Count} FP8 and {fp4Count} FP4 weights");

        IExecutionProvider provider = vulkan ? new VulkanProvider() : new CpuExecutionProvider();
        Assert.SkipUnless(provider.CheckAvailability().IsAvailable, "provider not available");
        var devices = provider.GetDevices();
        var info = vulkan ? devices.FirstOrDefault(d => d.Kind == DeviceKind.DiscreteGpu) ?? devices[0] : devices[0];
        int[] ids = [.. Enumerable.Range(0, Positions).Select(i => (i * 389 + 17) % 1000)];
        var expected = Run(dequantized, provider, info, ids);
        var actual = Run(quantized, provider, info, ids);
        double worst = 0;
        for (int p = 0; p < expected.Count; p++)
        {
            float max = expected[p].Max();
            float min = expected[p].Min();
            double difference = expected[p].Zip(actual[p], (a, b) => Math.Abs(a - b)).Max();
            worst = Math.Max(worst, difference / (max - min));
        }

        Assert.True(worst <= maxRelative, $"logits differ from the dequantized checkpoint by {worst:P3} of the spread (bound {maxRelative:P2})");
    }

    private static List<float[]> Run(string directory, IExecutionProvider provider, DeviceInfo info, int[] ids)
    {
        var slot = new DeviceSlot { Device = provider.OpenDevice(info) };
        try
        {
            using var model = LlmModel.Load(TestHost.Loader, directory, new LlmLoadOptions
            {
                Devices = [slot], ContextLength = 256, MaxBatchTokens = 64, KvCacheType = Sharplight.Tensors.DataType.F32,
                DisabledFeatures = new HashSet<string> { DeviceFeatures.IntDot },
            });
            using var session = model.CreateSession(WellKnown.DecodeGraph);
            var logits = new List<float[]> { session.Process(ids.AsSpan(0, PrefillChunk)).ToArray() };
            for (int i = PrefillChunk; i < ids.Length; i++)
            {
                logits.Add(session.Process(ids.AsSpan(i, 1)).ToArray());
            }

            return logits;
        }
        finally
        {
            slot.Device.Dispose();
        }
    }

    /// <summary>Writes the quantized checkpoint and its F32 dequantization next to each other; returns their directories.</summary>
    private (string Quantized, string Dequantized, int Fp8, int Fp4) Build(string source)
    {
        string quantized = Path.Combine(_root, "quantized");
        string dequantized = Path.Combine(_root, "dequantized");
        foreach (string directory in (string[])[quantized, dequantized])
        {
            Directory.CreateDirectory(directory);
            foreach (string file in Directory.GetFiles(source).Where(f => !f.EndsWith(".safetensors", StringComparison.Ordinal)))
            {
                File.Copy(file, Path.Combine(directory, Path.GetFileName(file)));
            }
        }

        var config = JsonNode(File.ReadAllText(Path.Combine(source, "config.json")));
        config["expert_dtype"] = "fp4";
        config["quantization_config"] = new System.Text.Json.Nodes.JsonObject
        {
            ["activation_scheme"] = "dynamic", ["fmt"] = "e4m3", ["quant_method"] = "fp8", ["scale_fmt"] = "ue8m0",
            ["weight_block_size"] = new System.Text.Json.Nodes.JsonArray(128, 128),
        };
        File.WriteAllText(Path.Combine(quantized, "config.json"), config.ToJsonString());

        var quantizedTensors = new List<(string, string, long[], byte[])>();
        var dequantizedTensors = new List<(string, string, long[], byte[])>();
        int fp8 = 0, fp4 = 0;
        foreach (var (name, dtype, shape, data) in ReadSafetensors(Path.Combine(source, "model.safetensors")))
        {
            if (dtype == "F32" && shape.Length == 2 && Fp8Weight().IsMatch(name))
            {
                var (codes, scales, scaleShape, values) = QuantizeFp8(Floats(data), shape[0], shape[1]);
                quantizedTensors.Add((name, "F8_E4M3", shape, codes));
                quantizedTensors.Add((name[..^".weight".Length] + ".scale", "F8_E8M0", scaleShape, scales));
                dequantizedTensors.Add((name, "F32", shape, Bytes(values)));
                fp8++;
            }
            else if (dtype == "F32" && shape.Length == 2 && shape[1] % 32 == 0 && Fp4Weight().IsMatch(name))
            {
                var (codes, scales, values) = QuantizeFp4(Floats(data));
                quantizedTensors.Add((name, "I8", [shape[0], shape[1] / 2], codes));
                quantizedTensors.Add((name[..^".weight".Length] + ".scale", "F8_E8M0", [shape[0], shape[1] / 32], scales));
                dequantizedTensors.Add((name, "F32", shape, Bytes(values)));
                fp4++;
            }
            else
            {
                quantizedTensors.Add((name, dtype, shape, data));
                dequantizedTensors.Add((name, dtype, shape, data));
            }
        }

        WriteSafetensors(Path.Combine(quantized, "model.safetensors"), quantizedTensors);
        WriteSafetensors(Path.Combine(dequantized, "model.safetensors"), dequantizedTensors);
        return (quantized, dequantized, fp8, fp4);
    }

    /// <summary>E4M3 codes with one E8M0 scale per 128 x 128 block (DeepSeek's act_quant with ue8m0 scales: 2^ceil(log2(amax / 448))).</summary>
    private static (byte[] Codes, byte[] Scales, long[] ScaleShape, float[] Values) QuantizeFp8(float[] weight, long rows, long columns)
    {
        long blockRows = (rows + 127) / 128;
        long blockColumns = (columns + 127) / 128;
        var codes = new byte[weight.Length];
        var scales = new byte[blockRows * blockColumns];
        var values = new float[weight.Length];
        for (long br = 0; br < blockRows; br++)
        {
            for (long bc = 0; bc < blockColumns; bc++)
            {
                float amax = 1e-4f;
                ForBlock(br, bc, i => amax = Math.Max(amax, Math.Abs(weight[i])));
                int exponent = (int)Math.Ceiling(Math.Log2(amax / 448.0));
                scales[br * blockColumns + bc] = (byte)(exponent + 127);
                float scale = MathF.ScaleB(1f, exponent);
                ForBlock(br, bc, i =>
                {
                    byte code = Nearest(s_e4m3, Math.Abs(weight[i]) / scale);
                    codes[i] = (byte)(code | (weight[i] < 0 ? 0x80 : 0));
                    values[i] = MathF.CopySign(s_e4m3[code] * scale, weight[i] < 0 ? -1f : 1f);
                });
            }
        }

        return (codes, scales, [blockRows, blockColumns], values);

        void ForBlock(long br, long bc, Action<long> action)
        {
            for (long r = br * 128; r < Math.Min(rows, br * 128 + 128); r++)
            {
                for (long c = bc * 128; c < Math.Min(columns, bc * 128 + 128); c++)
                {
                    action(r * columns + c);
                }
            }
        }
    }

    /// <summary>E2M1 codes packed two per byte (value 2i low) with one E8M0 scale per 32 values (2^ceil(log2(amax / 6))).</summary>
    private static (byte[] Codes, byte[] Scales, float[] Values) QuantizeFp4(float[] weight)
    {
        var codes = new byte[weight.Length / 2];
        var scales = new byte[weight.Length / 32];
        var values = new float[weight.Length];
        for (int block = 0; block < scales.Length; block++)
        {
            float amax = 1e-4f;
            for (int j = 0; j < 32; j++)
            {
                amax = Math.Max(amax, Math.Abs(weight[block * 32 + j]));
            }

            int exponent = (int)Math.Ceiling(Math.Log2(amax / 6.0));
            scales[block] = (byte)(exponent + 127);
            float scale = MathF.ScaleB(1f, exponent);
            for (int j = 0; j < 32; j++)
            {
                int i = block * 32 + j;
                byte magnitude = Nearest(s_e2m1, Math.Abs(weight[i]) / scale);
                int code = magnitude | (weight[i] < 0 ? 8 : 0);
                codes[i / 2] |= (byte)(code << (i % 2 * 4));
                values[i] = (weight[i] < 0 ? -1f : 1f) * s_e2m1[magnitude] * scale;
            }
        }

        return (codes, scales, values);
    }

    private static byte Nearest(float[] table, float value)
    {
        int best = 0;
        for (int code = 1; code < table.Length; code++)
        {
            if (Math.Abs(table[code] - value) < Math.Abs(table[best] - value))
            {
                best = code;
            }
        }

        return (byte)best;
    }

    private static float E4m3(byte code)
    {
        int exponent = (code >> 3) & 0xF;
        int mantissa = code & 7;
        return exponent == 0 ? MathF.ScaleB(mantissa / 8f, -6) : MathF.ScaleB(1f + mantissa / 8f, exponent - 7);
    }

    private static System.Text.Json.Nodes.JsonObject JsonNode(string json) => System.Text.Json.Nodes.JsonNode.Parse(json)!.AsObject();

    private static float[] Floats(byte[] data) => [.. Enumerable.Range(0, data.Length / 4).Select(i => BinaryPrimitives.ReadSingleLittleEndian(data.AsSpan(i * 4)))];

    private static byte[] Bytes(float[] values) => [.. values.SelectMany(BitConverter.GetBytes)];

    private static List<(string Name, string DType, long[] Shape, byte[] Data)> ReadSafetensors(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        long headerLength = BinaryPrimitives.ReadInt64LittleEndian(bytes);
        using var header = JsonDocument.Parse(bytes.AsMemory(8, (int)headerLength));
        var tensors = new List<(string, string, long[], byte[])>();
        foreach (var entry in header.RootElement.EnumerateObject().Where(e => e.Name != "__metadata__"))
        {
            var offsets = entry.Value.GetProperty("data_offsets");
            long begin = 8 + headerLength + offsets[0].GetInt64();
            long end = 8 + headerLength + offsets[1].GetInt64();
            tensors.Add((entry.Name, entry.Value.GetProperty("dtype").GetString()!,
                [.. entry.Value.GetProperty("shape").EnumerateArray().Select(d => d.GetInt64())], bytes[(int)begin..(int)end]));
        }

        return tensors;
    }

    private static void WriteSafetensors(string path, List<(string Name, string DType, long[] Shape, byte[] Data)> tensors)
    {
        var header = new Dictionary<string, object>();
        long offset = 0;
        foreach (var (name, dtype, shape, data) in tensors)
        {
            header[name] = new Dictionary<string, object> { ["dtype"] = dtype, ["shape"] = shape, ["data_offsets"] = new[] { offset, offset + data.Length } };
            offset += data.Length;
        }

        var json = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(header));
        using var stream = File.Create(path);
        stream.Write(BitConverter.GetBytes((ulong)json.Length));
        stream.Write(json);
        foreach (var tensor in tensors)
        {
            stream.Write(tensor.Data);
        }
    }
}
