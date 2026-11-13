using Sharplight.Llm;
using Sharplight.Runtime;

namespace Sharplight.Vision.Tests;

/// <summary>Vision encoder numerics against the Hugging Face transformers reference (torch, f32) on the Qwen3.5-0.8B tower.</summary>
public class VisionParityTests(ITestOutputHelper output)
{
    private static LlmModel LoadHf(DeviceSlot slot, ImageTokenLimits? limits = null)
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
        Assert.True(rms < 2e-2, $"relative RMS {rms}");   // GPUs may run bf16-weight matmuls with bf16 operands (~5e-3)
    }

    /// <summary>A non-square, larger image (resized by the preprocessor, 1200 patches): Vulkan vs CPU.</summary>
    [Fact]
    public void Vulkan_matches_cpu_on_a_large_non_square_image()
    {
        Assert.SkipUnless(Directory.Exists(Env.Hf08), "Qwen3.5-0.8B (HF) not available");
        var slot = Env.Vulkan();
        Assert.SkipUnless(slot is not null, "no Vulkan device");
        int w = Environment.GetEnvironmentVariable("SHARPLIGHT_TEST_W") is { } sw ? int.Parse(sw) : 640, h = Environment.GetEnvironmentVariable("SHARPLIGHT_TEST_H") is { } sh ? int.Parse(sh) : 488;
        var rgb = new byte[w * h * 3];
        var rng = new Random(7);
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                int o = (y * w + x) * 3;
                rgb[o] = (byte)(x * 255 / w); rgb[o + 1] = (byte)(y * 255 / h); rgb[o + 2] = (byte)(((x / 40 + y / 40) & 1) * 200 + rng.Next(40));
            }
        }

        var image = new ImageInput(w, h, rgb);
        float[] cpu, vk;
        using (var m = LoadHf(Env.Cpu())) { using var e = m.CreateVisionEncoder(); cpu = e.Encode(image); }
        using (var m = LoadHf(slot!)) { using var e = m.CreateVisionEncoder(); vk = e.Encode(image); }
        var (abs, rel, rms) = Env.Compare(vk, cpu);
        Console.Error.WriteLine($"Vulkan vs CPU ({w}x{h}): max abs {abs:E3}, max rel {rel:E3}, relative RMS {rms:E3}");
        Assert.True(rms < 2e-2, $"relative RMS {rms}");   // GPUs may run bf16-weight matmuls with bf16 operands (~5e-3)
    }
}
