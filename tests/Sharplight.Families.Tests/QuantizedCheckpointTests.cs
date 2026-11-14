using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Text.Json;
using Sharplight.Cpu;
using Sharplight.HuggingFace;
using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Vulkan;

namespace Sharplight.Families.Tests;

/// <summary>
/// Quantized Hugging Face checkpoints (GPTQ, AWQ, compressed-tensors NVFP4 / MXFP4 / integer, bitsandbytes, ModelOpt NVFP4; the
/// tiny ones of tools/mk_tiny_quantized.py and small real Hub checkpoints) end to end against transformers: the logits of
/// hf_quant_reference.py (an f32 forward over the library's own dequantization of the weights, activations not quantized) must be
/// matched on the CPU and on Vulkan. Tiny models are teacher-forced through one prefill chunk (tensor-core GEMMs on Vulkan) and then
/// token by token (matvecs), comparing every position; real models compare the last position of the prompt. The bound is relative
/// to the reference's logit spread, and the top-1 token must agree (or be within <see cref="NearTie"/> of the spread). Skips when
/// the checkpoints are absent.
/// </summary>
public class QuantizedCheckpointTests(ITestOutputHelper output)
{
    private const int PrefillChunk = 17;
    private const double NearTie = 0.02;

    private static readonly string Home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

    static QuantizedCheckpointTests()
    {
        HuggingFaceImporter.Register();
    }

    public static TheoryData<string> Checkpoints() =>
    [
        "tiny/tiny-llama-ct-nvfp4a16", "tiny/tiny-llama-ct-mxfp4a16", "tiny/tiny-llama-ct-w4a16-g128", "tiny/tiny-llama-ct-w4a16-actorder",
        "tiny/tiny-llama-ct-w4a16-asym-g32", "tiny/tiny-llama-ct-w8a16-channel", "tiny/tiny-llama-ct-w8a8-int", "tiny/tiny-llama-awq-gemm",
        "tiny/tiny-llama-awq-gemv", "tiny/tiny-llama-bnb-nf4", "tiny/tiny-llama-bnb-nf4-double", "tiny/tiny-llama-bnb-fp4",
        "tiny/tiny-llama-bnb-int8", "tiny/tiny-llama-gptq-4bit-g128-sym", "tiny/tiny-llama-gptq-4bit-g128-actorder",
        "tiny/tiny-llama-gptq-4bit-g32-v2", "tiny/tiny-llama-gptq-3bit-g128", "tiny/tiny-llama-gptq-8bit-channel", "tiny/tiny-llama-quark-mxfp4",
        "tiny/tiny-llama-mlx-4bit-g64", "tiny/tiny-llama-mlx-4bit-g32", "tiny/tiny-llama-mlx-8bit-g64",
        "NVFP4_Qwen3-0.6B-FP4", "mgoin_Qwen3-0.6B-NVFP4",
    ];

    [Theory]
    [MemberData(nameof(Checkpoints))]
    public void MatchesTransformersOnCpu(string checkpoint) => Run(checkpoint, vulkan: false, maxRelative: 0.01);

    [Theory]
    [MemberData(nameof(Checkpoints))]
    public void MatchesTransformersOnVulkan(string checkpoint) => Run(checkpoint, vulkan: true, maxRelative: 0.04);

    private void Run(string checkpoint, bool vulkan, double maxRelative)
    {
        string directory = Path.Combine(Home, "models", "hf", "quant", checkpoint);
        string referencePath = Path.Combine(directory, "sharplight-reference", "reference.safetensors");
        Assert.SkipUnless(File.Exists(referencePath), $"{referencePath} not present");
        var reference = ReadReference(referencePath);
        int[] ids = reference.Ids;
        int vocabulary = reference.Shape[1];
        int positions = reference.Shape[0];

        IExecutionProvider provider = vulkan ? new VulkanProvider() : new CpuExecutionProvider();
        Assert.SkipUnless(provider.CheckAvailability().IsAvailable, "provider not available");
        var devices = provider.GetDevices();
        var info = vulkan ? devices.FirstOrDefault(d => d.Kind == DeviceKind.DiscreteGpu) ?? devices[0] : devices[0];
        var slot = new DeviceSlot { Device = provider.OpenDevice(info) };
        try
        {
            // The CPU run keeps f32 activations (no 8-bit integer-dot matvec), so it shows the weight decode alone.
            var disabled = vulkan ? new HashSet<string>() : new HashSet<string> { DeviceFeatures.IntDot };
            using var model = LlmModel.Load(directory, new LlmLoadOptions
            {
                Devices = [slot], ContextLength = 256, MaxBatchTokens = 64, DisabledFeatures = disabled,
            });
            using var session = model.CreateSession(WellKnown.DecodeGraph);
            var chunks = new List<(int End, float[] Logits)>();
            if (positions == 1)
            {
                chunks.Add((ids.Length, [.. session.Process(ids)]));
            }
            else
            {
                chunks.Add((PrefillChunk, [.. session.Process(ids.AsSpan(0, PrefillChunk))]));
                for (int i = PrefillChunk; i < ids.Length; i++)
                {
                    chunks.Add((i + 1, [.. session.Process(ids.AsSpan(i, 1))]));
                }
            }

            double worst = 0;
            foreach (var (end, logits) in chunks)
            {
                var expected = reference.Logits.AsSpan((positions == 1 ? 0 : end - 1) * vocabulary, vocabulary);
                Assert.Equal(vocabulary, logits.Length);
                float max = float.MinValue;
                float min = float.MaxValue;
                int top = 0;
                int actualTop = 0;
                double difference = 0;
                for (int v = 0; v < vocabulary; v++)
                {
                    max = Math.Max(max, expected[v]);
                    min = Math.Min(min, expected[v]);
                    top = expected[v] > expected[top] ? v : top;
                    actualTop = logits[v] > logits[actualTop] ? v : actualTop;
                    difference = Math.Max(difference, Math.Abs(logits[v] - expected[v]));
                }

                double spread = max - min;
                worst = Math.Max(worst, difference / spread);
                Assert.True((expected[top] - expected[actualTop]) / spread <= NearTie,
                    $"{checkpoint} position {end - 1}: top-1 {actualTop}, transformers {top} ({expected[top] - expected[actualTop]:F3} apart, spread {spread:F2})");
            }

            output.WriteLine($"{checkpoint} {(vulkan ? "vulkan" : "cpu")}: {chunks.Count} positions, max logit difference {worst:P2} of the spread");
            Assert.True(worst <= maxRelative, $"{checkpoint}: logits differ by {worst:P2} of the spread (bound {maxRelative:P0})");
        }
        finally
        {
            slot.Device.Dispose();
        }
    }

    /// <summary>The ids (I32) and logits (F32 [positions, vocabulary]) of a reference file.</summary>
    private static (int[] Ids, float[] Logits, int[] Shape) ReadReference(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        long headerLength = BinaryPrimitives.ReadInt64LittleEndian(bytes);
        using var header = JsonDocument.Parse(bytes.AsMemory(8, (int)headerLength));
        ReadOnlySpan<byte> Data(string name, out int[] shape)
        {
            var entry = header.RootElement.GetProperty(name);
            shape = [.. entry.GetProperty("shape").EnumerateArray().Select(d => d.GetInt32())];
            var offsets = entry.GetProperty("data_offsets");
            long begin = 8 + headerLength + offsets[0].GetInt64();
            return bytes.AsSpan((int)begin, (int)(offsets[1].GetInt64() - offsets[0].GetInt64()));
        }

        int[] ids = MemoryMarshal.Cast<byte, int>(Data("ids", out _)).ToArray();
        float[] logits = MemoryMarshal.Cast<byte, float>(Data("logits", out int[] logitShape)).ToArray();
        return (ids, logits, logitShape);
    }
}
