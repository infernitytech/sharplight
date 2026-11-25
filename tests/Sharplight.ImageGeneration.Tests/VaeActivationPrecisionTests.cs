using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;

namespace Sharplight.ImageGeneration.Tests;

/// <summary>
/// The activation-range calibration of VAE decoders under reduced precision (<see cref="TaskModel.CalibrateActivationPrecision"/>, as the
/// image generator runs it): SDXL's decoder, whose activations exceed f16's range, gets bf16 activations and then decodes the diffusers
/// reference latents (~/models/sd/sdxl-reference) correctly; SD 1.5's and FLUX.1's keep f16. Skips when a model or a GPU is missing.
/// </summary>
[Collection(LargeModelCollection.Name)]
public sealed class VaeActivationPrecisionTests
{
    private static readonly Dictionary<string, long> s_probeSize = new() { [DiffusionGraphs.Height] = 256, [DiffusionGraphs.Width] = 256 };

    private static TaskModel? LoadReduced(string path)
    {
        var slot = Env.Vulkan();
        return slot is null || !Directory.Exists(path) ? null
            : TaskModel.Load(TestHost.Loader, new TaskLoadOptions { ModelPath = path, Devices = [slot], AllowReducedPrecision = true });
    }

    private static ActivationPrecision Calibrate(TaskModel model)
        => model.CalibrateActivationPrecision(ModelComponentRoles.Vae, DiffusionGraphs.VaeDecode, s_probeSize, (probe, parameters) =>
        {
            parameters.Symbols[DiffusionGraphs.BatchSymbol] = 1;
            var latent = probe.Graph.Inputs.Single(i => i.Name == DiffusionGraphs.Latent);
            probe.SetInput<float>(latent, LatentNoise.Draw(5, 0, (int)latent.Shape.Resolve(parameters.Symbols).Aggregate(1L, (a, b) => a * b)));
        });

    public static TheoryData<string, string> Decoders => new()
    {
        { "sdxl", nameof(ActivationPrecision.BFloat16) },
        { "sd15", nameof(ActivationPrecision.Half) },
        { "flux-schnell", nameof(ActivationPrecision.Half) },
    };

    [Theory]
    [MemberData(nameof(Decoders))]
    public void DecoderPrecisionFollowsItsActivationRange(string name, string expected)
    {
        string path = name switch
        {
            "sdxl" => Env.SdxlDiffusers,
            "sd15" => Env.Sd15Diffusers,
            _ => Env.FluxSchnellDiffusers,
        };
        using var model = LoadReduced(path);
        Assert.SkipWhen(model is null, $"{name} or a Vulkan GPU missing");
        Assert.Equal(expected, Calibrate(model!).ToString());
        Assert.Equal(expected, model!.ActivationPrecisionOf(DiffusionGraphs.VaeDecode).ToString());
    }

    [Fact]
    public void SdxlDecoderMatchesDiffusersUnderReducedPrecision()
    {
        string reference = Path.Combine(Path.GetDirectoryName(Env.SdxlDiffusers)!, "sdxl-reference");
        using var model = File.Exists(Path.Combine(reference, "vae_latent.f32")) ? LoadReduced(Env.SdxlDiffusers) : null;
        Assert.SkipWhen(model is null, "SDXL, its references or a Vulkan GPU missing");
        Assert.Equal(ActivationPrecision.BFloat16, Calibrate(model!));
        const float scaling = 0.13025f;
        var latent = Env.ReadFloats(Path.Combine(reference, "vae_latent.f32")).Select(v => v * scaling).ToArray();
        var decoded = GraphRunner.Run(model!, ModelComponentRoles.Vae, DiffusionGraphs.VaeDecode, DiffusionGraphs.Image, 1, (DiffusionGraphs.Latent, latent));
        var expected = Env.ReadFloats(Path.Combine(reference, "vae_decoded.f32")).Select(v => Math.Clamp(v / 2 + 0.5f, 0f, 1f)).ToArray();
        var (maxAbs, rms) = Env.Compare(decoded, expected);
        TestContext.Current.SendDiagnosticMessage($"vae (bf16 activations): relative RMS {rms:E2}, max {maxAbs:E2}");
        Assert.True(rms < 2e-2, $"vae: max {maxAbs}, relative RMS {rms}");
    }
}
