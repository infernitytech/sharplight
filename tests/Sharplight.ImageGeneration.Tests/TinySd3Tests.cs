using System.Text.Json;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;
using Sharplight.Vision;

namespace Sharplight.ImageGeneration.Tests;

/// <summary>
/// The tiny random SD3 pipelines (tools/make-tiny-sd3.py; "sd3" without q/k norms, "sd35" with them and the second image-only
/// attention): both CLIP encoders (penultimate hidden states, projected pooled embeddings), the T5 encoder, the transformer, the VAE and
/// whole guided text-to-image runs against diffusers, on the CPU and on Vulkan, from the diffusers folder and from the transformer-only
/// files (original names, Q8_0 GGUF) with companions.
/// </summary>
public sealed class TinySd3Tests
{
    private static string Root(string variant) => Path.Combine(Env.TinySd3, variant);

    private static float[] Tensor(string variant, string name) => Env.ReadFloats(Path.Combine(Root(variant), "reference", name + ".f32"));

    private static JsonElement Index(string variant)
    {
        using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(Root(variant), "reference", "index.json")));
        return document.RootElement.Clone();
    }

    private static bool Present(string variant) => File.Exists(Path.Combine(Root(variant), "reference", "index.json"));

    /// <summary>The companion files of the transformer-only variants, by role (the diffusers folder's component files).</summary>
    internal static Dictionary<string, string> Companions(string variant)
    {
        string diffusers = Path.Combine(Root(variant), "diffusers");
        return new Dictionary<string, string>
        {
            [AuxiliaryRoles.TextEncoder] = Path.Combine(diffusers, "text_encoder", "model.safetensors"),
            [AuxiliaryRoles.SecondTextEncoder] = Path.Combine(diffusers, "text_encoder_2", "model.safetensors"),
            [AuxiliaryRoles.ThirdTextEncoder] = Path.Combine(diffusers, "text_encoder_3", "model.safetensors"),
            [AuxiliaryRoles.Vae] = Path.Combine(diffusers, "vae", "diffusion_pytorch_model.safetensors"),
        };
    }

    /// <summary>The model of a variant from a source kind: "diffusers", "single" (original-name safetensors) or "gguf" (Q8_0), with companions.</summary>
    private static TaskModel Load(string variant, string source, DeviceSlot slot)
    {
        if (source == "diffusers")
        {
            return Env.Load(Path.Combine(Root(variant), "diffusers"), slot);
        }

        string file = Path.Combine(Root(variant), "single", source == "gguf" ? "sd3-Q8_0.gguf" : "sd3.safetensors");
        return Env.Load(file, slot, Companions(variant));
    }

    public static TheoryData<string, string, string> Cases()
    {
        var data = new TheoryData<string, string, string>();
        foreach (string device in new[] { "cpu", "vulkan" })
        {
            foreach (string variant in new[] { "sd3", "sd35" })
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
        Assert.SkipUnless(Present(variant), "tiny-random-sd3 missing");
        var slot = Env.Device(device);
        Assert.SkipWhen(slot is null, "no Vulkan device");
        using var model = Load(variant, source, slot!);
        var index = Index(variant);
        int height = index.GetProperty("height").GetInt32();
        int width = index.GetProperty("width").GetInt32();
        var dims = new Dictionary<string, long> { [DiffusionGraphs.Height] = height, [DiffusionGraphs.Width] = width };
        var second = index.GetProperty("tokens_2").EnumerateObject().ToList();
        var third = index.GetProperty("tokens_3").EnumerateObject().ToList();
        var tokenizer = model.CreateTokenizer(ModelComponentRoles.Tokenizer);
        var thirdTokenizer = model.CreateTokenizer(DiffusionGraphs.ThirdTokenizerRole);
        int i = 0;
        foreach (var prompt in index.GetProperty("tokens").EnumerateObject())
        {
            int[] tokens = [.. prompt.Value.EnumerateArray().Select(t => t.GetInt32())];
            int[] secondTokens = [.. second[i].Value.EnumerateArray().Select(t => t.GetInt32())];
            int[] thirdTokens = [.. third[i].Value.EnumerateArray().Select(t => t.GetInt32())];
            Assert.Equal(tokens.TakeWhile(t => t != 49407), tokenizer.Encode(prompt.Name, addSpecial: true, parseSpecial: false).TakeWhile(t => t != 49407));
            Assert.Equal(thirdTokens.TakeWhile(t => t != 0), thirdTokenizer.Encode(prompt.Name, addSpecial: true, parseSpecial: false));
            foreach (var (graph, tokenIds, name) in new[] { (DiffusionGraphs.TextEncode, tokens, "text"), (DiffusionGraphs.TextEncode2, secondTokens, "text2") })
            {
                var pooled = GraphRunner.Run(model, ModelComponentRoles.TextEncoder, graph, DiffusionGraphs.Pooled, 0, (DiffusionGraphs.Tokens, tokenIds));
                var (maxAbs, rms) = Env.Compare(pooled, Tensor(variant, $"{name}_pooled_{i}"));
                Assert.True(rms < 1e-4, $"{name} pooled {i}: max {maxAbs}, relative RMS {rms}");
                var hidden = GraphRunner.Run(model, ModelComponentRoles.TextEncoder, graph, DiffusionGraphs.Hidden, 0, (DiffusionGraphs.Tokens, tokenIds));
                (maxAbs, rms) = Env.Compare(hidden, Tensor(variant, $"{name}_hidden_{i}"));
                Assert.True(rms < 1e-4, $"{name} hidden {i}: max {maxAbs}, relative RMS {rms}");
            }

            var t5 = GraphRunner.Run(model, DiffusionGraphs.ThirdTextEncoderRole, DiffusionGraphs.TextEncode3, DiffusionGraphs.Hidden, 0,
                (DiffusionGraphs.Tokens, thirdTokens));
            var (t5Max, t5Rms) = Env.Compare(t5, Tensor(variant, $"text3_hidden_{i}"));
            Assert.True(t5Rms < 1e-4, $"T5 {i}: max {t5Max}, relative RMS {t5Rms}");
            i++;
        }

        var prediction = GraphRunner.Run(model, ModelComponentRoles.Denoiser, DiffusionGraphs.Denoise, DiffusionGraphs.Prediction, 1, dims,
            (DiffusionGraphs.Latent, Tensor(variant, "transformer_latent")),
            (DiffusionGraphs.Timestep, Tensor(variant, "transformer_timestep")),
            (DiffusionGraphs.Context, Tensor(variant, "transformer_context")),
            (DiffusionGraphs.Pooled, Tensor(variant, "transformer_pooled")));
        var (transformerMax, transformerRms) = Env.Compare(prediction, Tensor(variant, source == "gguf" ? "transformer_q8_out" : "transformer_out"));
        // Q8_0 matrices: the backends' integer dot products quantize the activations as well (llama.cpp's convention).
        double tolerance = source == "gguf" ? 5e-3 : 1e-4;
        Assert.True(transformerRms < tolerance, $"transformer: max {transformerMax}, relative RMS {transformerRms}");

        const float scaling = 1.5305f;
        const float shift = 0.0609f;
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
        Assert.SkipUnless(Present(variant), "tiny-random-sd3 missing");
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

    [Fact]
    public async Task PackageOfATransformerOnlyFileHoldsTheCompanions()
    {
        Assert.SkipUnless(Present("sd35"), "tiny-random-sd3 missing");
        _ = Env.Services;
        string folder = Directory.CreateTempSubdirectory("sl-sd3-").FullName;
        string package = Path.Combine(folder, "sd3-tiny.slm");
        try
        {
            TestHost.Converter.Convert(Path.Combine(Root("sd35"), "single", "sd3-Q8_0.gguf"), package, auxiliary: Companions("sd35"));
            var info = Sharplight.Package.ModelPackageReader.Open(package);
            Assert.Equal(ModelKinds.ImageGeneration, info.Definition.Description.Kind);
            Assert.Equal(["text-encoder", "text-encoder-3", "tokenizer", "tokenizer-3", "denoiser", "vae", "scheduler"],
                info.Definition.Description.Components.Select(c => c.Role));
            using var model = Env.Load(package, Env.Cpu());
            using var generator = Env.Clients.Create<IImageGenerator>(model);
            var index = Index("sd35");
            int height = index.GetProperty("height").GetInt32();
            int width = index.GetProperty("width").GetInt32();
            var options = new ImageGenerationOptions
            {
                Width = width,
                Height = height,
                Steps = index.GetProperty("steps").GetInt32(),
                GuidanceScale = index.GetProperty("guidance_scale").GetSingle(),
                InitialNoise = Tensor("sd35", "noise"),
            };
            var result = await generator.GenerateAsync(new ImageGenerationRequest { Prompt = index.GetProperty("prompt").GetString()! }, options,
                TestContext.Current.CancellationToken);
            double psnr = Env.Psnr(result.Images[0].Rgb, PlanarImage.FromPlanar(Tensor("sd35", "q8_image"), width, height, 0f, 1f).Rgb);
            Assert.True(psnr > 40, $"package: PSNR {psnr:F1} dB");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }
}
