using Sharplight.Imaging;

namespace Sharplight.ImageGeneration.Tests;

/// <summary>
/// Whole text-to-image runs of the tiny pipeline against diffusers with the same initial noise, for every deterministic sampler,
/// plus the task-layer behaviour (seeds, streaming, previews, other sizes, img2img, inpainting).
/// </summary>
public sealed class TinyPipelineTests
{
    public static TheoryData<string, string> Runs()
    {
        var data = new TheoryData<string, string>();
        foreach (string device in new[] { "cpu", "vulkan" })
        {
            foreach (string sampler in new[] { Samplers.Euler, Samplers.EulerKarras, Samplers.Ddim, Samplers.Pndm, Samplers.DpmPlusPlus2M, Samplers.DpmPlusPlus2MKarras })
            {
                data.Add(device, sampler);
            }
        }

        return data;
    }

    [Theory]
    [MemberData(nameof(Runs))]
    public async Task TextToImageMatchesDiffusers(string device, string sampler)
    {
        Assert.SkipUnless(File.Exists(Path.Combine(Env.TinyReference, "noise.f32")), "tiny references missing");
        var slot = Env.Device(device);
        Assert.SkipWhen(slot is null, "no Vulkan device");
        using var model = Env.Load(Env.TinyDiffusers, slot!);
        using var generator = Env.Clients.Create<IImageGenerator>(model);
        var index = Env.TinyIndex();
        var request = new ImageGenerationRequest
        {
            Prompt = index.GetProperty("prompt").GetString()!,
            NegativePrompt = index.GetProperty("negative_prompt").GetString(),
        };
        var options = new ImageGenerationOptions
        {
            Width = index.GetProperty("width").GetInt32(),
            Height = index.GetProperty("height").GetInt32(),
            Steps = index.GetProperty("steps").GetInt32(),
            GuidanceScale = index.GetProperty("guidance_scale").GetSingle(),
            Scheduler = sampler,
            InitialNoise = Env.TinyTensor("noise"),
        };
        var result = await generator.GenerateAsync(request, options, TestContext.Current.CancellationToken);
        var expected = PlanarImage.FromPlanar(Env.TinyTensor(sampler + "_image"), options.Width!.Value, options.Height!.Value, 0f, 1f);
        double psnr = Env.Psnr(result.Images[0].Rgb, expected.Rgb);
        Assert.True(psnr > 40, $"{sampler} on {device}: PSNR {psnr:F1} dB");
    }

    [Fact]
    public async Task SeedsAreReproducibleAndStreamingReportsSteps()
    {
        Assert.SkipUnless(Directory.Exists(Env.Tiny), "tiny-random-sd missing");
        using var model = Env.Load(Env.TinyDiffusers, Env.Cpu());
        using var generator = Env.Clients.Create<IImageGenerator>(model);
        var request = new ImageGenerationRequest { Prompt = "a red cube", NegativePrompt = "blurry" };
        var options = new ImageGenerationOptions { Steps = 4, Seed = 7, Count = 2, Scheduler = Samplers.EulerAncestral, PreviewInterval = 2 };
        var updates = new List<ImageGenerationUpdate>();
        await foreach (var update in generator.GenerateStreamingAsync(request, options, TestContext.Current.CancellationToken))
        {
            updates.Add(update);
        }

        Assert.Equal(9, updates.Count);
        Assert.Equal(Enumerable.Range(1, 8), updates.Take(8).Select(u => u.Step));
        Assert.All(updates, u => Assert.Equal(8, u.TotalSteps));
        Assert.Equal(4, updates.Count(u => u.Preview is not null));
        var first = updates[^1].Result!;
        Assert.Equal(2, first.Images.Count);
        Assert.Equal(7, first.Seed);
        Assert.Equal(64, first.Images[0].Width);
        Assert.NotEqual(first.Images[0].Rgb, first.Images[1].Rgb);

        var again = await generator.GenerateAsync(request, options with { Count = 1 }, TestContext.Current.CancellationToken);
        Assert.Equal(first.Images[0].Rgb, again.Images[0].Rgb);
        var single = await generator.GenerateAsync(request, options with { Count = 1, Seed = 8 }, TestContext.Current.CancellationToken);
        Assert.Equal(first.Images[1].Rgb, single.Images[0].Rgb);
    }

    [Fact]
    public async Task OtherSizesImageToImageAndInpainting()
    {
        Assert.SkipUnless(Directory.Exists(Env.Tiny), "tiny-random-sd missing");
        using var model = Env.Load(Env.TinyDiffusers, Env.Cpu());
        using var generator = Env.Clients.Create<IImageGenerator>(model);
        var ct = TestContext.Current.CancellationToken;
        var wide = await generator.GenerateAsync(new ImageGenerationRequest { Prompt = "a landscape" },
            new ImageGenerationOptions { Width = 128, Height = 64, Steps = 3, Seed = 1 }, ct);
        Assert.Equal(128, wide.Images[0].Width);
        Assert.Equal(64, wide.Images[0].Height);

        var source = wide.Images[0];
        var options = new ImageGenerationOptions { Steps = 4, Seed = 3, Scheduler = Samplers.Ddim };
        // Strength 0 runs no step: the VAE round trip of the input (a random VAE does not reconstruct it).
        var roundTrip = (await generator.GenerateAsync(new ImageGenerationRequest { Prompt = "a landscape", Image = source, Strength = 0f }, options, ct)).Images[0];
        var low = await generator.GenerateAsync(new ImageGenerationRequest { Prompt = "a landscape", Image = source, Strength = 0.25f }, options, ct);
        var high = await generator.GenerateAsync(new ImageGenerationRequest { Prompt = "a landscape", Image = source, Strength = 1f }, options, ct);
        Assert.Equal(source.Width, low.Images[0].Width);
        Assert.True(Env.Psnr(low.Images[0].Rgb, roundTrip.Rgb) > Env.Psnr(high.Images[0].Rgb, roundTrip.Rgb), "a lower strength keeps more of the input");

        // Keep the left half: the mask is white (regenerate) only on the right.
        var mask = new byte[source.Rgb.Length];
        for (int y = 0; y < source.Height; y++)
        {
            for (int x = source.Width / 2; x < source.Width; x++)
            {
                mask.AsSpan((y * source.Width + x) * 3, 3).Fill(255);
            }
        }

        var inpainted = await generator.GenerateAsync(new ImageGenerationRequest
        {
            Prompt = "a landscape",
            Image = source,
            Mask = new Vision.ImageInput(source.Width, source.Height, mask),
            Strength = 1f,
        }, options, ct);
        Assert.Equal(source.Width, inpainted.Images[0].Width);
    }
}
