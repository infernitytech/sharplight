using System.Text.Json;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tasks;
using Sharplight.Vision;

namespace Sharplight.ImageGeneration.Tests;

/// <summary>
/// The tiny random FLUX.2 Klein pipeline (tools/make-tiny-flux2.py): the chat-templated Qwen3 prompt tokens (thinking disabled), the text
/// encoder's hidden states entering layers 9, 18 and 27 concatenated per token, the transformer (shared modulations, SwiGLU, 4-axis RoPE),
/// the 2×2-patchified, batch-norm-normalized VAE (decode and encode) and a distilled text-to-image run against diffusers, on the CPU and on
/// Vulkan, from the diffusers folder and from the transformer-only files (original names, Q8_0 GGUF) with the folder's text encoder and
/// VAE as companions.
/// </summary>
public sealed class TinyFlux2Tests
{
    private static string Root => Env.TinyFlux2;

    private static float[] Tensor(string name) => Env.ReadFloats(Path.Combine(Root, "reference", name + ".f32"));

    private static JsonElement Index()
    {
        using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(Root, "reference", "index.json")));
        return document.RootElement.Clone();
    }

    private static bool Present => File.Exists(Path.Combine(Root, "reference", "index.json"));

    private static TaskModel Load(string source, DeviceSlot slot)
    {
        string diffusers = Path.Combine(Root, "diffusers");
        if (source == "diffusers")
        {
            return Env.Load(diffusers, slot);
        }

        return Env.Load(Path.Combine(Root, "single", source == "gguf" ? "flux2-Q8_0.gguf" : "flux2.safetensors"), slot, new Dictionary<string, string>
        {
            [AuxiliaryRoles.TextEncoder] = Path.Combine(diffusers, "text_encoder"),
            [AuxiliaryRoles.Vae] = Path.Combine(diffusers, "vae", "diffusion_pytorch_model.safetensors"),
        });
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
        Assert.SkipUnless(Present, "tiny-random-flux2 missing");
        var slot = Env.Device(device);
        Assert.SkipWhen(slot is null, "no Vulkan device");
        using var model = Load(source, slot!);
        var index = Index();
        int height = index.GetProperty("height").GetInt32();
        int width = index.GetProperty("width").GetInt32();
        var dims = new Dictionary<string, long> { [DiffusionGraphs.Height] = height, [DiffusionGraphs.Width] = width };
        int i = 0;
        foreach (var prompt in index.GetProperty("tokens").EnumerateObject())
        {
            int[] tokens = [.. prompt.Value.EnumerateArray().Select(t => t.GetInt32())];
            int length = index.GetProperty("lengths").GetProperty(prompt.Name).GetInt32();
            float[] mask = [.. Enumerable.Range(0, tokens.Length).Select(t => t < length ? 1f : 0f)];
            var hidden = GraphRunner.Run(model, ModelComponentRoles.TextEncoder, DiffusionGraphs.TextEncode, DiffusionGraphs.Hidden, 0, (DiffusionGraphs.Tokens, tokens),
                (DiffusionGraphs.TokenMask, mask));
            int d = hidden.Length / tokens.Length;
            // The pipeline uses every row, the padding included.
            var (promptMax, promptRms) = Env.Compare(hidden.AsSpan(0, length * d), Tensor($"text_hidden_{i}").AsSpan(0, length * d));
            TestContext.Current.SendDiagnosticMessage($"text encoder {i}: prompt rows {promptRms:E2}");
            var (maxAbs, rms) = Env.Compare(hidden, Tensor($"text_hidden_{i}"));
            Assert.True(rms < 1e-4, $"text encoder {i}: max {maxAbs}, relative RMS {rms}");
            i++;
        }

        var prediction = GraphRunner.Run(model, ModelComponentRoles.Denoiser, DiffusionGraphs.Denoise, DiffusionGraphs.Prediction, 1, dims,
            (DiffusionGraphs.Latent, Tensor("transformer_latent")),
            (DiffusionGraphs.Timestep, Tensor("transformer_timestep")),
            (DiffusionGraphs.Context, Tensor("transformer_context")));
        var (transformerMax, transformerRms) = Env.Compare(prediction, Tensor(source == "gguf" ? "transformer_q8_out" : "transformer_out"));
        // Q8_0 matrices: the backends' integer dot products quantize the activations as well; Vulkan's f16 attention products.
        double tolerance = source == "gguf" ? (device == "cpu" ? 3e-2 : 5e-3) : (device == "cpu" ? 1e-4 : 1e-3);
        Assert.True(transformerRms < tolerance, $"transformer: max {transformerMax}, relative RMS {transformerRms}");

        var decoded = GraphRunner.Run(model, ModelComponentRoles.Vae, DiffusionGraphs.VaeDecode, DiffusionGraphs.Image, 1, dims,
            (DiffusionGraphs.Latent, Tensor("vae_latent")));
        var (vaeMax, vaeRms) = Env.Compare(decoded, Tensor("vae_decoded").Select(v => Math.Clamp(v / 2 + 0.5f, 0f, 1f)).ToArray());
        Assert.True(vaeRms < 1e-4, $"vae decode: max {vaeMax}, relative RMS {vaeRms}");
        var encoded = GraphRunner.Run(model, ModelComponentRoles.Vae, DiffusionGraphs.VaeEncode, DiffusionGraphs.Latent, 1, dims,
            (DiffusionGraphs.Image, Tensor("vae_image")));
        var (encodeMax, encodeRms) = Env.Compare(encoded, Tensor("vae_encoded"));
        Assert.True(encodeRms < 1e-4, $"vae encode: max {encodeMax}, relative RMS {encodeRms}");
    }

    [Fact]
    public void PromptTokensFollowTheChatTemplate()
    {
        Assert.SkipUnless(Present, "tiny-random-flux2 missing");
        using var model = Load("diffusers", Env.Cpu());
        var index = Index();
        Assert.Equal(index.GetProperty("template").GetString(), model.Description.Component(ModelComponentRoles.Tokenizer).ConfigValue<string?>(DiffusionConfigKeys.PromptTemplate, null));
        var tokenizer = model.CreateTokenizer(ModelComponentRoles.Tokenizer);
        string template = index.GetProperty("template").GetString()!;
        int at = template.IndexOf("{prompt}", StringComparison.Ordinal);
        foreach (var prompt in index.GetProperty("tokens").EnumerateObject())
        {
            int length = index.GetProperty("lengths").GetProperty(prompt.Name).GetInt32();
            int[] expected = [.. prompt.Value.EnumerateArray().Select(t => t.GetInt32()).Take(length)];
            int[] actual = [.. tokenizer.Encode(template[..at], addSpecial: false, parseSpecial: true),
                .. tokenizer.Encode(prompt.Name, addSpecial: false, parseSpecial: false),
                .. tokenizer.Encode(template[(at + "{prompt}".Length)..], addSpecial: false, parseSpecial: true)];
            Assert.Equal(expected, actual);
            Assert.Equal(prompt.Value[length].GetInt32(), tokenizer.PadId);
        }
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public async Task TextToImageMatchesDiffusers(string device, string source)
    {
        Assert.SkipUnless(Present, "tiny-random-flux2 missing");
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
            InitialNoise = Tensor("noise"),
        };
        var result = await generator.GenerateAsync(new ImageGenerationRequest { Prompt = index.GetProperty("prompt").GetString()! }, options,
            TestContext.Current.CancellationToken);
        var expected = PlanarImage.FromPlanar(Tensor(source == "gguf" ? "q8_image" : "image"), width, height, 0f, 1f);
        double psnr = Env.Psnr(result.Images[0].Rgb, expected.Rgb);
        Assert.True(psnr > 40, $"FLUX.2 from {source} on {device}: PSNR {psnr:F1} dB");
    }
}
