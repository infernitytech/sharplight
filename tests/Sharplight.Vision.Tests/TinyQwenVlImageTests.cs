using System.Text.Json;
using Sharplight.Llm;
using Sharplight.Models;

namespace Sharplight.Vision.Tests;

/// <summary>
/// Tiny random Qwen2-VL (LayerNorm, quick-GELU tower), Qwen2.5-VL (RMS norms, gated MLP, windowed attention) and PaddleOCR-VL (SigLIP tower
/// with 2-D rotary embedding, post-norm and a projector merger; mk_tiny_paddleocr_vl.py) and GLM-4.1V (bicubic position grid, patch norm,
/// convolution merger and GLM projector; mk_tiny_glm4v.py) and HunyuanOCR (half-pixel bilinear position grid, perceiver projector with a newline
/// column and begin / end rows, XD-RoPE positions in the language model; mk_tiny_hunyuan_vl.py) and CogVLM (EVA2-CLIP tower with class token and
/// post-norm blocks, GLU projector, visual experts and grouped positions in the language model; mk_tiny_cogvlm.py, references from its remote code) checkpoints against
/// transformers: the tower's merged-token embeddings, the prompt logits of an image prompt and the greedy continuation, on the CPU and the GPU.
/// The checkpoints come from tests/Sharplight.Families.Tests/tools/mk_tiny_qwen2_vl.py, the references (reference_image.json next to the
/// weights) from tools/hf_image_reference.py, which feeds the same synthetic normalized image into the tower without resizing. The GGUF
/// variants run the f32 GGUF and mmproj files converted by llama.cpp (tools/convert_tiny_mmproj.py, convert_tiny_cogvlm.py) against the same
/// references: the image path end to end through the GGUF mappings (image token, projector, layout and positions from the mmproj).
/// </summary>
public class TinyQwenVlImageTests
{
    private static readonly string Home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

    /// <summary>The synthetic normalized image of hf_image_reference.py, interleaved RGB.</summary>
    internal static float[] SineImage(int width, int height)
    {
        var v = new float[width * height * 3];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                for (int c = 0; c < 3; c++)
                {
                    v[(y * width + x) * 3 + c] = (float)Math.Sin(0.031 * x + 0.017 * (c + 1) * y + 0.5 * c);
                }
            }
        }

        return v;
    }

    [Theory]
    [InlineData("tiny-random-qwen2-vl-wide", false)]
    [InlineData("tiny-random-qwen2-vl-wide", true)]
    [InlineData("tiny-random-qwen2-5-vl-wide", false)]
    [InlineData("tiny-random-qwen2-5-vl-wide", true)]
    [InlineData("tiny-random-paddleocr-vl", false)]
    [InlineData("tiny-random-paddleocr-vl", true)]
    [InlineData("tiny-random-glm4v", false)]
    [InlineData("tiny-random-glm4v", true)]
    [InlineData("tiny-random-hunyuan-vl", false)]
    [InlineData("tiny-random-hunyuan-vl", true)]
    [InlineData("tiny-random-cogvlm", false)]
    [InlineData("tiny-random-cogvlm", true)]
    public void Tiny_qwen_vl_image_prompt_matches_transformers(string name, bool gpu)
    {
        string dir = Path.Combine(Home, "models/hf", name);
        string refPath = Path.Combine(dir, "reference_image.json");
        Assert.SkipUnless(File.Exists(refPath), $"{name} or its reference_image.json not available");
        var slot = gpu ? Env.Vulkan() : Env.Cpu();
        Assert.SkipWhen(slot is null, "no Vulkan device");
        using var model = LlmModel.Load(dir, new LlmLoadOptions { Devices = [slot!], ContextLength = 256, ImageTokens = new ImageTokenLimits(null, 128) });
        CheckImagePrompt(model, refPath, name, gpu);
    }

    [Theory]
    [InlineData("tiny-random-qwen2-vl-wide", false)]
    [InlineData("tiny-random-qwen2-vl-wide", true)]
    [InlineData("tiny-random-qwen2-5-vl-wide", false)]
    [InlineData("tiny-random-qwen2-5-vl-wide", true)]
    [InlineData("tiny-random-paddleocr-vl", false)]
    [InlineData("tiny-random-paddleocr-vl", true)]
    [InlineData("tiny-random-glm4v", false)]
    [InlineData("tiny-random-glm4v", true)]
    [InlineData("tiny-random-hunyuan-vl", false)]
    [InlineData("tiny-random-hunyuan-vl", true)]
    [InlineData("tiny-random-cogvlm", false)]
    [InlineData("tiny-random-cogvlm", true)]
    public void Tiny_vl_gguf_mmproj_image_prompt_matches_transformers(string name, bool gpu)
    {
        string refPath = Path.Combine(Home, "models/hf", name, "reference_image.json");
        string gguf = Path.Combine(Home, "models/gguf", $"{name}-f32.gguf");
        string mmproj = Path.Combine(Home, "models/gguf", $"mmproj-{name}-f32.gguf");
        Assert.SkipUnless(File.Exists(refPath) && File.Exists(gguf) && File.Exists(mmproj), $"{name}: reference_image.json, GGUF or mmproj not available");
        var slot = gpu ? Env.Vulkan() : Env.Cpu();
        Assert.SkipWhen(slot is null, "no Vulkan device");
        using var model = LlmModel.Load(gguf, new LlmLoadOptions
        {
            Devices = [slot!], ContextLength = 256, Auxiliary = [(AuxiliaryRoles.Mmproj, mmproj)], ImageTokens = new ImageTokenLimits(null, 128),
        });
        CheckImagePrompt(model, refPath, name + " GGUF", gpu);
    }

    // The tower's embeddings, the prompt's top logits and the greedy continuation against reference_image.json.
    private static void CheckImagePrompt(LlmModel model, string refPath, string name, bool gpu)
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(refPath));
        var r = doc.RootElement;
        int width = r.GetProperty("width").GetInt32();
        int height = r.GetProperty("height").GetInt32();
        var expected = r.GetProperty("embeddings").EnumerateArray().Select(e => e.GetSingle()).ToArray();
        using var vision = model.CreateVisionEncoder();
        var image = vision.Preprocessor.Patchify(SineImage(width, height), width, height);
        Assert.Equal(r.GetProperty("image_tokens").GetInt32(), image.TokenCount);
        var embeddings = vision.Encode(image);
        var (maxAbs, _, rms) = Env.Compare(embeddings, expected);
        Console.Error.WriteLine($"{name} ({(gpu ? "Vulkan" : "CPU")}): embeddings max |diff| {maxAbs:E2}, relative rms {rms:E2}");
        // The GPU's reduced-precision accumulation leaves ~5e-4 relative rms on the GLM-4V and HunyuanVL towers (O(1) outputs after their norms).
        Assert.True(maxAbs < (gpu ? 3e-3f : 2e-4f), $"embeddings differ by {maxAbs}");

        var promptIds = r.GetProperty("prompt_ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        var mm = MultimodalRegistry.Find(model.Definition.Architecture)!.Assemble(new MultimodalContext(model.Definition, t => Single(model, t)), promptIds, [image], 0);
        using var session = model.CreateSession(WellKnown.DecodeMultimodalGraph);
        var logits = session.Process(mm, [embeddings]);
        var topIds = r.GetProperty("top_ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        var topLogits = r.GetProperty("top_logits").EnumerateArray().Select(e => e.GetSingle()).ToArray();
        for (int i = 0; i < 5; i++)
        {
            Assert.True(Math.Abs(logits[topIds[i]] - topLogits[i]) < (gpu ? 5e-2f : 1e-2f), $"logit {topIds[i]}: {logits[topIds[i]]} vs {topLogits[i]}");
        }

        var greedy = r.GetProperty("greedy").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        var ids = new List<int>();
        int next = Enumerable.Range(0, logits.Length).MaxBy(i => logits[i]);
        while (ids.Count < greedy.Length)
        {
            ids.Add(next);
            var l = session.Process([next]);
            next = Enumerable.Range(0, l.Length).MaxBy(i => l[i]);
        }

        Assert.Equal(greedy, ids.ToArray());
    }

    private static int? Single(LlmModel model, string text)
    {
        var ids = model.Tokenizer.Encode(text, addSpecial: false, parseSpecial: true);
        return ids.Length == 1 ? ids[0] : null;
    }
}
