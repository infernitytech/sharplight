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
/// The tiny Gemma 3 (4×4 average pooling, soft-embedding norm, framed image rows) and Idefics3 / SmolVLM (tiles plus overview, pixel shuffle,
/// the processor's prompt layout) and LFM2-VL (SigLIP 2 NaFlex tower on tiles and a thumbnail of its own size, antialiased position grid, pixel
/// unshuffle with LayerNorm and MLP, the LFM2 hybrid language model) checkpoints come from mk_tiny_siglip_vl.py, their references from
/// siglip_vl_reference.py (GGUF variants only: the Hugging Face importer does not read their towers).
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
        using var model = LlmModel.Load(TestHost.Loader, dir, new LlmLoadOptions { Devices = [slot!], ContextLength = 256, ImageTokens = new ImageTokenLimits(null, 128) });
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
    [InlineData("tiny-random-gemma3-vl", false)]
    [InlineData("tiny-random-gemma3-vl", true)]
    [InlineData("tiny-random-idefics3", false)]
    [InlineData("tiny-random-idefics3", true)]
    [InlineData("tiny-random-lfm2-vl", false)]
    [InlineData("tiny-random-lfm2-vl", true)]
    public void Tiny_vl_gguf_mmproj_image_prompt_matches_transformers(string name, bool gpu)
    {
        string refPath = Path.Combine(Home, "models/hf", name, "reference_image.json");
        string gguf = Path.Combine(Home, "models/gguf", $"{name}-f32.gguf");
        string mmproj = Path.Combine(Home, "models/gguf", $"mmproj-{name}-f32.gguf");
        Assert.SkipUnless(File.Exists(refPath) && File.Exists(gguf) && File.Exists(mmproj), $"{name}: reference_image.json, GGUF or mmproj not available");
        var slot = gpu ? Env.Vulkan() : Env.Cpu();
        Assert.SkipWhen(slot is null, "no Vulkan device");
        using var model = LlmModel.Load(TestHost.Loader, gguf, new LlmLoadOptions
        {
            Devices = [slot!], ContextLength = 256, Auxiliary = { [AuxiliaryRoles.Mmproj] = mmproj }, ImageTokens = new ImageTokenLimits(null, 128),
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
        var image = r.TryGetProperty("thumbnail_width", out var thumbnailWidth)
            ? ViewsImage(vision.Preprocessor, width, height, r.GetProperty("tile_columns").GetInt32(), r.GetProperty("tile_rows").GetInt32(),
                thumbnailWidth.GetInt32(), r.GetProperty("thumbnail_height").GetInt32())
            : r.TryGetProperty("tile_columns", out var tileColumns)
            ? TiledImage(vision.Preprocessor, width, height, tileColumns.GetInt32(), r.GetProperty("tile_rows").GetInt32())
            : vision.Preprocessor.Patchify(SineImage(width, height), width, height);
        Assert.Equal(r.GetProperty("image_tokens").GetInt32(), image.TokenCount);
        var embeddings = vision.Encode(image);
        var (maxAbs, _, rms) = Env.Compare(embeddings, expected);
        Console.Error.WriteLine($"{name} ({(gpu ? "Vulkan" : "CPU")}): embeddings max |diff| {maxAbs:E2}, relative rms {rms:E2}");
        // The GPU's reduced-precision accumulation leaves ~5e-4 relative rms on the GLM-4V and HunyuanVL towers (O(1) outputs after their norms).
        Assert.True(maxAbs < (gpu ? 3e-3f : 2e-4f), $"embeddings differ by {maxAbs}");

        var promptIds = r.GetProperty("prompt_ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        var mm = Multimodal(model).Assemble(new MultimodalContext(model.Definition, t => Single(model, t)), promptIds, [image], 0);
        if (r.TryGetProperty("expanded_ids", out var expandedIds))
        {
            // The processor's own prompt layout: the text rows equal its tokens, the image rows sit where it puts image tokens.
            var hf = expandedIds.EnumerateArray().Select(e => e.GetInt32()).ToArray();
            Assert.Equal(hf.Length, mm.Length);
            int imageId = hf.GroupBy(t => t).MaxBy(g => g.Count())!.Key;
            for (int i = 0; i < hf.Length; i++)
            {
                Assert.Equal(hf[i], mm.ImageIndex[i] >= 0 ? imageId : mm.Tokens[i]);
            }
        }

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

    /// <summary>
    /// The views of siglip_vl_reference.py's tiled image: the tiles of a <paramref name="columns"/> × <paramref name="rows"/> grid cut from the sine
    /// image of the whole canvas and an overview holding the negated sine pattern of one tile.
    /// </summary>
    private static PreprocessedImage TiledImage(VisionPreprocessor pre, int width, int height, int columns, int rows)
    {
        int tile = width / columns;
        var canvas = SineImage(width, height);
        var tiles = new List<float[]>();
        for (int ty = 0; ty < rows; ty++)
        {
            for (int tx = 0; tx < columns; tx++)
            {
                var view = new float[3 * tile * tile];
                for (int y = 0; y < tile; y++)
                {
                    Array.Copy(canvas, 3 * ((ty * tile + y) * width + tx * tile), view, 3 * y * tile, 3 * tile);
                }

                tiles.Add(view);
            }
        }

        var overview = SineImage(tile, tile).Select(v => -v).ToArray();
        return pre.PatchifyTiles(overview, tiles, columns, rows);
    }

    /// <summary>
    /// The views of siglip_vl_reference.py's LFM2-VL image: the tiles of a <paramref name="columns"/> × <paramref name="rows"/> grid cut from the sine
    /// image of the whole canvas and a thumbnail of its own size holding the negated sine pattern.
    /// </summary>
    private static PreprocessedImage ViewsImage(VisionPreprocessor pre, int width, int height, int columns, int rows, int thumbnailWidth, int thumbnailHeight)
    {
        int tile = width / columns;
        var canvas = SineImage(width, height);
        var views = new List<(float[], int, int)>();
        for (int ty = 0; ty < rows; ty++)
        {
            for (int tx = 0; tx < columns; tx++)
            {
                var view = new float[3 * tile * tile];
                for (int y = 0; y < tile; y++)
                {
                    Array.Copy(canvas, 3 * ((ty * tile + y) * width + tx * tile), view, 3 * y * tile, 3 * tile);
                }

                views.Add((view, tile, tile));
            }
        }

        views.Add((SineImage(thumbnailWidth, thumbnailHeight).Select(v => -v).ToArray(), thumbnailWidth, thumbnailHeight));
        return pre.PatchifyViews(views, columns, rows);
    }

    private static int? Single(LlmModel model, string text)
    {
        var ids = model.Tokenizer.Encode(text, addSpecial: false, parseSpecial: true);
        return ids.Length == 1 ? ids[0] : null;
    }

    private static IMultimodalArchitecture Multimodal(LlmModel model)
        => model.Architectures.TryFindMultimodal(model.Definition.Architecture, out var multimodal)
            ? multimodal
            : throw new InvalidOperationException($"No multimodal prompt assembly for {model.Definition.Architecture}.");
}
