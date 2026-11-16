using System.Text.Json;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;
using Sharplight.Vision;

namespace Sharplight.ImageGeneration.Tests;

/// <summary>
/// The tiny random Chroma pipeline (tools/make-tiny-chroma.py): the masked T5 encoder, the transformer with the modulation
/// approximator and the masked text attention, the VAE and whole guided text-to-image runs against diffusers (with the boolean
/// transformer mask of the original Chroma), on the CPU and on Vulkan, from the diffusers folder and from the transformer-only files
/// (original names, Q8_0 GGUF) with companions.
/// </summary>
public sealed class TinyChromaTests
{
    private static string Root => Env.TinyChroma;

    private static float[] Tensor(string name) => Env.ReadFloats(Path.Combine(Root, "reference", name + ".f32"));

    private static JsonElement Index()
    {
        using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(Root, "reference", "index.json")));
        return document.RootElement.Clone();
    }

    private static bool Present => File.Exists(Path.Combine(Root, "reference", "index.json"));

    /// <summary>The companion files of the transformer-only files, by role.</summary>
    private static Dictionary<string, string> Companions()
    {
        string diffusers = Path.Combine(Root, "diffusers");
        return new Dictionary<string, string>
        {
            [AuxiliaryRoles.TextEncoder] = Path.Combine(diffusers, "text_encoder", "model.safetensors"),
            [AuxiliaryRoles.Vae] = Path.Combine(diffusers, "vae", "diffusion_pytorch_model.safetensors"),
        };
    }

    private static TaskModel Load(string source, DeviceSlot slot)
    {
        if (source == "diffusers")
        {
            return Env.Load(Path.Combine(Root, "diffusers"), slot);
        }

        string file = Path.Combine(Root, "single", source == "gguf" ? "chroma-Q8_0.gguf" : "chroma.safetensors");
        return Env.Load(file, slot, Companions());
    }

    public static TheoryData<string, string> Cases()
    {
        var data = new TheoryData<string, string>();
        foreach (string device in new[] { "cpu", "vulkan" })
        {
            foreach (string source in new[] { "diffusers", "single", "gguf" })
            {
                data.Add(device, source);
            }
        }

        return data;
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void GraphsMatchDiffusers(string device, string source)
    {
        Assert.SkipUnless(Present, "tiny-random-chroma missing");
        var slot = Env.Device(device);
        Assert.SkipWhen(slot is null, "no Vulkan device");
        using var model = Load(source, slot!);
        var index = Index();
        int height = index.GetProperty("height").GetInt32();
        int width = index.GetProperty("width").GetInt32();
        var dims = new Dictionary<string, long> { [DiffusionGraphs.Height] = height, [DiffusionGraphs.Width] = width };
        var tokenizer = model.CreateTokenizer(ModelComponentRoles.Tokenizer);
        int i = 0;
        foreach (var prompt in index.GetProperty("tokens").EnumerateObject())
        {
            int[] tokens = [.. prompt.Value.EnumerateArray().Select(t => t.GetInt32())];
            int length = tokens.TakeWhile(t => t != 0).Count();
            Assert.Equal(tokens.Take(length), tokenizer.Encode(prompt.Name, addSpecial: true, parseSpecial: false));
            var mask = new float[tokens.Length];
            mask.AsSpan(0, length).Fill(1f);
            var hidden = GraphRunner.Run(model, ModelComponentRoles.TextEncoder, DiffusionGraphs.TextEncode, DiffusionGraphs.Hidden, 0,
                (DiffusionGraphs.Tokens, tokens), (DiffusionGraphs.TokenMask, mask));
            // The padding rows are computed but never seen (the transformer masks them); compare the prompt's rows.
            int d = hidden.Length / tokens.Length;
            var (maxAbs, rms) = Env.Compare(hidden.AsSpan(0, length * d), Tensor($"text_hidden_{i}").AsSpan(0, length * d));
            Assert.True(rms < 1e-4, $"T5 {i}: max {maxAbs}, relative RMS {rms}");
            i++;
        }

        var prediction = GraphRunner.Run(model, ModelComponentRoles.Denoiser, DiffusionGraphs.Denoise, DiffusionGraphs.Prediction, 1, dims,
            (DiffusionGraphs.Latent, Tensor("transformer_latent")),
            (DiffusionGraphs.Timestep, Tensor("transformer_timestep")),
            (DiffusionGraphs.Context, Tensor("transformer_context")),
            (DiffusionGraphs.ContextMask, Tensor("transformer_context_mask")));
        var (transformerMax, transformerRms) = Env.Compare(prediction, Tensor(source == "gguf" ? "transformer_q8_out" : "transformer_out"));
        double tolerance = source == "gguf" ? 5e-3 : 1e-4;
        Assert.True(transformerRms < tolerance, $"transformer: max {transformerMax}, relative RMS {transformerRms}");

        const float scaling = 0.3611f;
        const float shift = 0.1159f;
        var decoded = GraphRunner.Run(model, ModelComponentRoles.Vae, DiffusionGraphs.VaeDecode, DiffusionGraphs.Image, 1, dims,
            (DiffusionGraphs.Latent, Tensor("vae_latent").Select(v => (v - shift) * scaling).ToArray()));
        var (vaeMax, vaeRms) = Env.Compare(decoded, Tensor("vae_decoded").Select(v => Math.Clamp(v / 2 + 0.5f, 0f, 1f)).ToArray());
        Assert.True(vaeRms < 1e-4, $"vae decode: max {vaeMax}, relative RMS {vaeRms}");
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public async Task TextToImageMatchesDiffusers(string device, string source)
    {
        Assert.SkipUnless(Present, "tiny-random-chroma missing");
        var slot = Env.Device(device);
        Assert.SkipWhen(slot is null, "no Vulkan device");
        using var model = Load(source, slot!);
        using var generator = Env.Clients.Create<IImageGenerator>(model);
        var index = Index();
        int height = index.GetProperty("height").GetInt32();
        int width = index.GetProperty("width").GetInt32();
        var options = new ImageGenerationOptions
        {
            Width = width,
            Height = height,
            Steps = index.GetProperty("steps").GetInt32(),
            GuidanceScale = index.GetProperty("guidance_scale").GetSingle(),
            InitialNoise = Tensor("noise"),
        };
        var request = new ImageGenerationRequest
        {
            Prompt = index.GetProperty("prompt").GetString()!,
            NegativePrompt = index.GetProperty("negative_prompt").GetString()!,
        };
        var result = await generator.GenerateAsync(request, options, TestContext.Current.CancellationToken);
        var expected = PlanarImage.FromPlanar(Tensor(source == "gguf" ? "q8_image" : "image"), width, height, 0f, 1f);
        double psnr = Env.Psnr(result.Images[0].Rgb, expected.Rgb);
        Assert.True(psnr > 40, $"Chroma from {source} on {device}: PSNR {psnr:F1} dB");
    }
}
