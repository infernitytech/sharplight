using Sharplight.Llm;
using Sharplight.Runtime;
using Sharplight.Vision;
using Xunit.Sdk;

namespace Sharplight.Vision.Tests;

/// <summary>Vision encoder numerics against the Hugging Face transformers reference (torch, f32) on the Qwen3.5-0.8B tower.</summary>
public class VisionParityTests(ITestOutputHelper output)
{
    static LlmModel LoadHf(DeviceSlot slot, ImageTokenLimits? limits = null)
        => LlmModel.Load(Env.Hf08, new LlmLoadOptions { Devices = [slot], ContextLength = 256, MaxBatchTokens = 64, ImageTokens = limits });

    [Fact]
    public void Cpu_embeddings_match_torch()
    {
        Assert.SkipUnless(Directory.Exists(Env.Hf08), "Qwen3.5-0.8B (HF) not available");
        using var model = LoadHf(Env.Cpu());
        using var enc = model.CreateVisionEncoder();
        var got = enc.Encode(Env.Shapes());
        var expected = Env.ReadFloats(Env.Fixture("shapes_hf_embeddings_100x1024.f32"));
        var (abs, rel, rms) = Env.Compare(got, expected);
        { output.WriteLine($"CPU vs torch: max abs {abs:E3}, max rel {rel:E3}, relative RMS {rms:E3}"); Console.Error.WriteLine($"CPU vs torch: max abs {abs:E3}, max rel {rel:E3}, relative RMS {rms:E3}"); }
        Assert.True(rms < 1e-4, $"relative RMS {rms}");
    }

    [Fact]
    public void Vulkan_embeddings_match_torch()
    {
        Assert.SkipUnless(Directory.Exists(Env.Hf08), "Qwen3.5-0.8B (HF) not available");
        var slot = Env.Vulkan();
        Assert.SkipUnless(slot is not null, "no Vulkan device");
        using var model = LoadHf(slot!);
        using var enc = model.CreateVisionEncoder();
        var got = enc.Encode(Env.Shapes());
        var expected = Env.ReadFloats(Env.Fixture("shapes_hf_embeddings_100x1024.f32"));
        var (abs, rel, rms) = Env.Compare(got, expected);
        { output.WriteLine($"Vulkan vs torch: max abs {abs:E3}, max rel {rel:E3}, relative RMS {rms:E3}"); Console.Error.WriteLine($"Vulkan vs torch: max abs {abs:E3}, max rel {rel:E3}, relative RMS {rms:E3}"); }
        Assert.True(rms < 1e-3, $"relative RMS {rms}");
    }
}
