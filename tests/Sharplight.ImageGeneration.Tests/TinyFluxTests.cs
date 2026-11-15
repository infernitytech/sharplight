using System.Text.Json;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;
using Sharplight.Vision;

namespace Sharplight.ImageGeneration.Tests;

/// <summary>
/// The tiny random FLUX pipelines (tools/make-tiny-flux.py; "dev" with the guidance embedding and the dynamic shift, "schnell"
/// without): the CLIP pooled embedding, the T5 encoder, the transformer, the VAE and whole text-to-image runs against diffusers, on
/// the CPU and on Vulkan, from the diffusers folder and from the transformer-only files (original names, Q8_0 GGUF) with companions.
/// </summary>
public sealed class TinyFluxTests
{
    private static string Root(string variant) => Path.Combine(Env.TinyFlux, variant);

    private static float[] Tensor(string variant, string name) => Env.ReadFloats(Path.Combine(Root(variant), "reference", name + ".f32"));

    private static JsonElement Index(string variant)
    {
        using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(Root(variant), "reference", "index.json")));
        return document.RootElement.Clone();
    }

    private static bool Present(string variant) => File.Exists(Path.Combine(Root(variant), "reference", "index.json"));

    /// <summary>The model of a variant from a source kind: "diffusers", "single" (original-name safetensors) or "gguf" (Q8_0), with companions.</summary>
    private static TaskModel Load(string variant, string source, DeviceSlot slot)
    {
        string diffusers = Path.Combine(Root(variant), "diffusers");
        if (source == "diffusers")
        {
            return Env.Load(diffusers, slot);
        }

        string file = Path.Combine(Root(variant), "single", source == "gguf" ? "flux-Q8_0.gguf" : "flux.safetensors");
        return Env.Load(file, slot, new Dictionary<string, string>
        {
            [AuxiliaryRoles.TextEncoder] = Path.Combine(diffusers, "text_encoder", "model.safetensors"),
            [AuxiliaryRoles.SecondTextEncoder] = Path.Combine(diffusers, "text_encoder_2", "model.safetensors"),
            [AuxiliaryRoles.Vae] = Path.Combine(diffusers, "vae", "diffusion_pytorch_model.safetensors"),
        });
    }

    public static TheoryData<string, string, string> Cases()
    {
        var data = new TheoryData<string, string, string>();
        foreach (string device in new[] { "cpu", "vulkan" })
        {
            foreach (string variant in new[] { "dev", "schnell" })
            {
                foreach (string source in new[] { "diffusers", "single", "gguf" })
                {
                    data.Add(device, variant, source);
                }
            }
        }

        return data;
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void GraphsMatchDiffusers(string device, string variant, string source)
    {
        Assert.SkipUnless(Present(variant), "tiny-random-flux missing");
        var slot = Env.Device(device);
        Assert.SkipWhen(slot is null, "no Vulkan device");
        using var model = Load(variant, source, slot!);
        var index = Index(variant);
        int height = index.GetProperty("height").GetInt32();
        int width = index.GetProperty("width").GetInt32();
        var dims = new Dictionary<string, long> { [DiffusionGraphs.Height] = height, [DiffusionGraphs.Width] = width };
        var second = index.GetProperty("tokens_2").EnumerateObject().ToList();
        var tokenizer = model.CreateTokenizer(ModelComponentRoles.Tokenizer);
        var secondTokenizer = model.CreateTokenizer(DiffusionGraphs.SecondTokenizerRole);
        int i = 0;
        foreach (var prompt in index.GetProperty("tokens").EnumerateObject())
        {
            int[] tokens = [.. prompt.Value.EnumerateArray().Select(t => t.GetInt32())];
            int[] secondTokens = [.. second[i].Value.EnumerateArray().Select(t => t.GetInt32())];
            Assert.Equal(tokens.TakeWhile(t => t != 49407), tokenizer.Encode(prompt.Name, addSpecial: true, parseSpecial: false).TakeWhile(t => t != 49407));
            Assert.Equal(secondTokens.TakeWhile(t => t != 0), secondTokenizer.Encode(prompt.Name, addSpecial: true, parseSpecial: false));
            var pooled = GraphRunner.Run(model, ModelComponentRoles.TextEncoder, DiffusionGraphs.TextEncode, DiffusionGraphs.Pooled, 0, (DiffusionGraphs.Tokens, tokens));
            var (maxAbs, rms) = Env.Compare(pooled, Tensor(variant, $"text_pooled_{i}"));
            Assert.True(rms < 1e-4, $"pooled {i}: max {maxAbs}, relative RMS {rms}");
            var hidden = GraphRunner.Run(model, DiffusionGraphs.SecondTextEncoderRole, DiffusionGraphs.TextEncode2, DiffusionGraphs.Hidden, 0,
                (DiffusionGraphs.Tokens, secondTokens));
            (maxAbs, rms) = Env.Compare(hidden, Tensor(variant, $"text2_hidden_{i}"));
            Assert.True(rms < 1e-4, $"T5 {i}: max {maxAbs}, relative RMS {rms}");
            i++;
        }

        var inputs = new List<(string, Array)>
        {
            (DiffusionGraphs.Latent, Tensor(variant, "transformer_latent")),
            (DiffusionGraphs.Timestep, Tensor(variant, "transformer_timestep")),
            (DiffusionGraphs.Context, Tensor(variant, "transformer_context")),
            (DiffusionGraphs.Pooled, Tensor(variant, "transformer_pooled")),
        };
        if (variant == "dev")
        {
            inputs.Add((DiffusionGraphs.Guidance, Tensor(variant, "transformer_guidance")));
        }

        var prediction = GraphRunner.Run(model, ModelComponentRoles.Denoiser, DiffusionGraphs.Denoise, DiffusionGraphs.Prediction, 1, dims, [.. inputs]);
        var (transformerMax, transformerRms) = Env.Compare(prediction, Tensor(variant, source == "gguf" ? "transformer_q8_out" : "transformer_out"));
        Assert.True(transformerRms < 1e-4, $"transformer: max {transformerMax}, relative RMS {transformerRms}");

        const float scaling = 0.3611f;
        const float shift = 0.1159f;
        var decoded = GraphRunner.Run(model, ModelComponentRoles.Vae, DiffusionGraphs.VaeDecode, DiffusionGraphs.Image, 1, dims,
            (DiffusionGraphs.Latent, Tensor(variant, "vae_latent").Select(v => (v - shift) * scaling).ToArray()));
        var (vaeMax, vaeRms) = Env.Compare(decoded, Tensor(variant, "vae_decoded").Select(v => Math.Clamp(v / 2 + 0.5f, 0f, 1f)).ToArray());
        Assert.True(vaeRms < 1e-4, $"vae decode: max {vaeMax}, relative RMS {vaeRms}");
        var encoded = GraphRunner.Run(model, ModelComponentRoles.Vae, DiffusionGraphs.VaeEncode, DiffusionGraphs.Latent, 1, dims,
            (DiffusionGraphs.Image, Tensor(variant, "vae_image").Select(v => v / 2 + 0.5f).ToArray()));
        var (encodeMax, encodeRms) = Env.Compare(encoded, Tensor(variant, "vae_encoded_mean").Select(v => (v - shift) * scaling).ToArray());
        Assert.True(encodeRms < 1e-4, $"vae encode: max {encodeMax}, relative RMS {encodeRms}");
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public async Task TextToImageMatchesDiffusers(string device, string variant, string source)
    {
        Assert.SkipUnless(Present(variant), "tiny-random-flux missing");
        var slot = Env.Device(device);
        Assert.SkipWhen(slot is null, "no Vulkan device");
        using var model = Load(variant, source, slot!);
        using var generator = Env.Clients.Create<IImageGenerator>(model);
        var index = Index(variant);
        int height = index.GetProperty("height").GetInt32();
        int width = index.GetProperty("width").GetInt32();
        var options = new ImageGenerationOptions
        {
            Width = width,
            Height = height,
            Steps = index.GetProperty("steps").GetInt32(),
            GuidanceScale = index.GetProperty("guidance_scale").GetSingle(),
            InitialNoise = Tensor(variant, "noise"),
        };
        var request = new ImageGenerationRequest { Prompt = index.GetProperty("prompt").GetString()! };
        var result = await generator.GenerateAsync(request, options, TestContext.Current.CancellationToken);
        var expected = PlanarImage.FromPlanar(Tensor(variant, source == "gguf" ? "q8_image" : "image"), width, height, 0f, 1f);
        double psnr = Env.Psnr(result.Images[0].Rgb, expected.Rgb);
        Assert.True(psnr > 40, $"{variant} from {source} on {device}: PSNR {psnr:F1} dB");
    }
}
