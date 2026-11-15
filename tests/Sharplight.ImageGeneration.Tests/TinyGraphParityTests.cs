using Sharplight.Models;

namespace Sharplight.ImageGeneration.Tests;

/// <summary>
/// The tiny random SD 1.x and 2.x pipelines' text encoder, UNet and VAE against the diffusers / transformers reference tensors, on the CPU
/// and on Vulkan.
/// </summary>
public sealed class TinyGraphParityTests
{
    public static TheoryData<string, string> Runs()
    {
        var data = new TheoryData<string, string>();
        foreach (string family in new[] { "sd1", "sd2" })
        {
            foreach (string device in new[] { "cpu", "vulkan" })
            {
                data.Add(family, device);
            }
        }

        return data;
    }

    [Theory]
    [MemberData(nameof(Runs))]
    public void TextEncoderMatchesTransformers(string family, string device)
    {
        Assert.SkipUnless(Directory.Exists(Env.TinyRoot(family)), $"tiny {family} pipeline missing");
        var slot = Env.Device(device);
        Assert.SkipWhen(slot is null, "no Vulkan device");
        using var model = Env.Load(Env.TinyDiffusersOf(family), slot!);
        var index = Env.TinyIndexOf(family);
        int i = 0;
        foreach (var prompt in index.GetProperty("tokens").EnumerateObject())
        {
            var tokens = prompt.Value.EnumerateArray().Select(t => t.GetInt32()).ToArray();
            var hidden = GraphRunner.Text(model, tokens);
            var (maxAbs, rms) = Env.Compare(hidden, Env.TinyTensorOf(family, $"text_hidden_{i}"));
            Assert.True(rms < 1e-4, $"prompt {i}: max {maxAbs}, relative RMS {rms}");
            i++;
        }
    }

    [Theory]
    [MemberData(nameof(Runs))]
    public void UNetMatchesDiffusers(string family, string device)
    {
        Assert.SkipUnless(Directory.Exists(Env.TinyRoot(family)), $"tiny {family} pipeline missing");
        var slot = Env.Device(device);
        Assert.SkipWhen(slot is null, "no Vulkan device");
        using var model = Env.Load(Env.TinyDiffusersOf(family), slot!);
        var output = GraphRunner.Run(model, ModelComponentRoles.Denoiser, DiffusionGraphs.Denoise, DiffusionGraphs.Prediction, 2,
            (DiffusionGraphs.Latent, Env.TinyTensorOf(family, "unet_latent")), (DiffusionGraphs.Timestep, Env.TinyTensorOf(family, "unet_timesteps")),
            (DiffusionGraphs.Context, Env.TinyTensorOf(family, "unet_context")));
        var (maxAbs, rms) = Env.Compare(output, Env.TinyTensorOf(family, "unet_out"));
        Assert.True(rms < 1e-4, $"max {maxAbs}, relative RMS {rms}");
    }

    [Theory]
    [MemberData(nameof(Runs))]
    public void VaeMatchesDiffusers(string family, string device)
    {
        Assert.SkipUnless(Directory.Exists(Env.TinyRoot(family)), $"tiny {family} pipeline missing");
        var slot = Env.Device(device);
        Assert.SkipWhen(slot is null, "no Vulkan device");
        using var model = Env.Load(Env.TinyDiffusersOf(family), slot!);
        // The graphs take model latents (scaled) and images in [0, 1]; the references are raw VAE latents and images in [-1, 1].
        const float scaling = 0.18215f;
        var latent = Env.TinyTensorOf(family, "vae_latent").Select(v => v * scaling).ToArray();
        var decoded = GraphRunner.Run(model, ModelComponentRoles.Vae, DiffusionGraphs.VaeDecode, DiffusionGraphs.Image, 1, (DiffusionGraphs.Latent, latent));
        var expectedImage = Env.TinyTensorOf(family, "vae_decoded").Select(v => Math.Clamp(v / 2 + 0.5f, 0f, 1f)).ToArray();
        var (maxAbs, rms) = Env.Compare(decoded, expectedImage);
        Assert.True(rms < 1e-4, $"decode: max {maxAbs}, relative RMS {rms}");

        var image = Env.TinyTensorOf(family, "vae_image").Select(v => v / 2 + 0.5f).ToArray();
        var encoded = GraphRunner.Run(model, ModelComponentRoles.Vae, DiffusionGraphs.VaeEncode, DiffusionGraphs.Latent, 1, (DiffusionGraphs.Image, image));
        var expectedLatent = Env.TinyTensorOf(family, "vae_encoded_mean").Select(v => v * scaling).ToArray();
        (maxAbs, rms) = Env.Compare(encoded, expectedLatent);
        Assert.True(rms < 1e-4, $"encode: max {maxAbs}, relative RMS {rms}");
    }
}
