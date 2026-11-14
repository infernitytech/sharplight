using System.Text.Json;
using Sharplight.Cpu;
using Sharplight.Gguf;
using Sharplight.HuggingFace;
using Sharplight.Llm;
using Sharplight.Runtime;
using Sharplight.Vulkan;

namespace Sharplight.Families.Tests;

/// <summary>
/// Masked-diffusion decoding (Dream, LLaDA) of tiny random checkpoints against the greedy references <c>tools/mk_tiny_diffusion.py</c> writes
/// next to them (<c>reference_diffusion.json</c>: transformers' forward pass with an all-visible mask and the reference unmasking loops). The
/// GGUF conversions (<c>tools/convert_tiny_gguf.py</c>) are checked against the same reference. Cases skip when the files are absent.
/// </summary>
public class DiffusionGenerationTests
{
    private static readonly string Home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

    static DiffusionGenerationTests()
    {
        GgufImporter.Register();
        HuggingFaceImporter.Register();
    }

    /// <summary>Model path under ~/models and the directory holding its reference.</summary>
    public static TheoryData<string, string, bool> Cases { get; } = new()
    {
        { "hf/tiny-random-dream", "hf/tiny-random-dream", false },
        { "hf/tiny-random-dream", "hf/tiny-random-dream", true },
        { "gguf/tiny-random-dream-f32.gguf", "hf/tiny-random-dream", false },
        { "gguf/tiny-random-dream-f32.gguf", "hf/tiny-random-dream", true },
        { "hf/tiny-random-llada", "hf/tiny-random-llada", false },
        { "hf/tiny-random-llada", "hf/tiny-random-llada", true },
        { "gguf/tiny-random-llada-f32.gguf", "hf/tiny-random-llada", false },
        { "gguf/tiny-random-llada-f32.gguf", "hf/tiny-random-llada", true },
    };

    /// <summary>The generated ids equal the reference's.</summary>
    [Theory]
    [MemberData(nameof(Cases))]
    public void MatchesReference(string model, string referenceDir, bool vulkan)
    {
        string path = Path.Combine(Home, "models", model);
        string referencePath = Path.Combine(Home, "models", referenceDir, "reference_diffusion.json");
        if ((!File.Exists(path) && !Directory.Exists(path)) || !File.Exists(referencePath))
        {
            Assert.Skip($"{path} or its reference not present");
        }

        using var json = JsonDocument.Parse(File.ReadAllText(referencePath));
        var r = json.RootElement;
        int[] prompt = r.GetProperty("prompt").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        int[] expected = r.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        var options = new GenerationOptions
        {
            MaxNewTokens = r.GetProperty("span").GetInt32(),
            StopAtEog = false,
            Diffusion = new DiffusionOptions
            {
                Steps = r.GetProperty("steps").GetInt32(),
                BlockLength = r.GetProperty("block_length").GetInt32(),
                Eps = r.GetProperty("eps").GetSingle(),
            },
        };

        DeviceSlot slot;
        if (vulkan)
        {
            var p = new VulkanProvider();
            if (!p.CheckAvailability().IsAvailable)
            {
                Assert.Skip("Vulkan is not available");
            }

            var devs = p.GetDevices();
            var dev = devs.FirstOrDefault(d => d.Kind == DeviceKind.DiscreteGpu) ?? devs.FirstOrDefault();
            if (dev is null)
            {
                Assert.Skip("no Vulkan device");
            }

            slot = new DeviceSlot { Device = p.OpenDevice(dev) };
        }
        else
        {
            var p = new CpuExecutionProvider();
            slot = new DeviceSlot { Device = p.OpenDevice(p.GetDevices()[0]) };
        }

        try
        {
            using var llm = LlmModel.Load(path, new LlmLoadOptions { Devices = [slot], ContextLength = 64, MaxBatchTokens = 64 });
            Assert.True(llm.IsDiffusion);
            using var generator = llm.CreateDiffusionGenerator();
            var ids = generator.Generate(prompt, options);
            Assert.Equal(expected, ids);
        }
        finally
        {
            slot.Device.Dispose();
        }
    }
}
