using Sharplight.Models;

namespace Sharplight.ImageGeneration.Tests;

/// <summary>
/// The tiny random SD 1.x pipeline's text encoder, UNet and VAE against the diffusers / transformers reference tensors, on the CPU
/// and on Vulkan.
/// </summary>
public sealed class TinyGraphParityTests
{
    public static TheoryData<string> Devices => ["cpu", "vulkan"];

    [Theory]
    [MemberData(nameof(Devices))]
    public void TextEncoderMatchesTransformers(string device)
    {
        Assert.SkipUnless(Directory.Exists(Env.Tiny), "tiny-random-sd missing");
        var slot = Env.Device(device);
        Assert.SkipWhen(slot is null, "no Vulkan device");
        using var model = Env.Load(Env.TinyDiffusers, slot!);
        var index = Env.TinyIndex();
        int i = 0;
        foreach (var prompt in index.GetProperty("tokens").EnumerateObject())
        {
            var tokens = prompt.Value.EnumerateArray().Select(t => t.GetInt32()).ToArray();
            var hidden = GraphRunner.Text(model, tokens);
            var (maxAbs, rms) = Env.Compare(hidden, Env.TinyTensor($"text_hidden_{i}"));
            Assert.True(rms < 1e-4, $"prompt {i}: max {maxAbs}, relative RMS {rms}");
            i++;
        }
    }

    [Theory]
    [MemberData(nameof(Devices))]
    public void UNetMatchesDiffusers(string device)
    {
        Assert.SkipUnless(Directory.Exists(Env.Tiny), "tiny-random-sd missing");
        var slot = Env.Device(device);
        Assert.SkipWhen(slot is null, "no Vulkan device");
        using var model = Env.Load(Env.TinyDiffusers, slot!);
        var output = GraphRunner.Run(model, ModelComponentRoles.Denoiser, DiffusionGraphs.Denoise, DiffusionGraphs.Prediction, 2,
            (DiffusionGraphs.Latent, Env.TinyTensor("unet_latent")), (DiffusionGraphs.Timestep, Env.TinyTensor("unet_timesteps")),
            (DiffusionGraphs.Context, Env.TinyTensor("unet_context")));
        var (maxAbs, rms) = Env.Compare(output, Env.TinyTensor("unet_out"));
        Assert.True(rms < 1e-4, $"max {maxAbs}, relative RMS {rms}");
    }

    [Theory]
    [MemberData(nameof(Devices))]
    public void VaeMatchesDiffusers(string device)
    {
        Assert.SkipUnless(Directory.Exists(Env.Tiny), "tiny-random-sd missing");
        var slot = Env.Device(device);
        Assert.SkipWhen(slot is null, "no Vulkan device");
        using var model = Env.Load(Env.TinyDiffusers, slot!);
        // The graphs take model latents (scaled) and images in [0, 1]; the references are raw VAE latents and images in [-1, 1].
        const float scaling = 0.18215f;
        var latent = Env.TinyTensor("vae_latent").Select(v => v * scaling).ToArray();
        var decoded = GraphRunner.Run(model, ModelComponentRoles.Vae, DiffusionGraphs.VaeDecode, DiffusionGraphs.Image, 1, (DiffusionGraphs.Latent, latent));
        var expectedImage = Env.TinyTensor("vae_decoded").Select(v => Math.Clamp(v / 2 + 0.5f, 0f, 1f)).ToArray();
        var (maxAbs, rms) = Env.Compare(decoded, expectedImage);
        Assert.True(rms < 1e-4, $"decode: max {maxAbs}, relative RMS {rms}");

        var image = Env.TinyTensor("vae_image").Select(v => v / 2 + 0.5f).ToArray();
        var encoded = GraphRunner.Run(model, ModelComponentRoles.Vae, DiffusionGraphs.VaeEncode, DiffusionGraphs.Latent, 1, (DiffusionGraphs.Image, image));
        var expectedLatent = Env.TinyTensor("vae_encoded_mean").Select(v => v * scaling).ToArray();
        (maxAbs, rms) = Env.Compare(encoded, expectedLatent);
        Assert.True(rms < 1e-4, $"encode: max {maxAbs}, relative RMS {rms}");
    }
}
