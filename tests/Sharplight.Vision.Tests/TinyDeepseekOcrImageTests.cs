using System.Text.Json;
using Sharplight.Llm;
using Sharplight.Models;

namespace Sharplight.Vision.Tests;

/// <summary>
/// The two-stage image tower of DeepSeek-OCR and Unlimited-OCR (SAM ViT-B, convolution neck, CLIP over the SAM features, projector, newline and
/// view-separator layout) on tiny random checkpoints (tests/Sharplight.Families.Tests/tools/mk_tiny_deepseek_ocr.py) against transformers
/// (tools/hf_ocr_image_reference.py, which feeds synthetic normalized views into the model without the image processor): the image rows the
/// language model receives, the prompt logits of an image prompt and the greedy continuation, on the CPU and on Vulkan. The global view alone
/// uses reference_image_global.json, the global view with a 3 × 2 grid of tiles reference_image.json. tiny-random-deepseek-ocr has the
/// Unlimited-OCR weights in DeepSeek-OCR's remote-code config (whose layout code is the same), so it is checked against the Unlimited-OCR files.
/// </summary>
public class TinyDeepseekOcrImageTests
{
    private static readonly string Home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

    [Theory]
    [InlineData("tiny-random-unlimited-ocr", "reference_image_global.json", false)]
    [InlineData("tiny-random-unlimited-ocr", "reference_image_global.json", true)]
    [InlineData("tiny-random-unlimited-ocr", "reference_image.json", false)]
    [InlineData("tiny-random-unlimited-ocr", "reference_image.json", true)]
    [InlineData("tiny-random-deepseek-ocr", "reference_image.json", false)]
    [InlineData("tiny-random-deepseek-ocr", "reference_image.json", true)]
    public void Tiny_deepseek_ocr_image_prompt_matches_transformers(string name, string reference, bool gpu)
    {
        string dir = Path.Combine(Home, "models/hf", name);
        string refPath = Path.Combine(Home, "models/hf/tiny-random-unlimited-ocr", reference);
        Assert.SkipUnless(Directory.Exists(dir) && File.Exists(refPath), $"{name} or {refPath} not available");
        var slot = gpu ? Env.Vulkan() : Env.Cpu();
        Assert.SkipWhen(slot is null, "no Vulkan device");
        using var doc = JsonDocument.Parse(File.ReadAllBytes(refPath));
        var r = doc.RootElement;
        int size = r.GetProperty("size").GetInt32();
        int tile = r.GetProperty("tile_size").GetInt32();
        int columns = r.GetProperty("columns").GetInt32();
        int rows = r.GetProperty("rows").GetInt32();
        var expected = r.GetProperty("embeddings").EnumerateArray().Select(e => e.GetSingle()).ToArray();

        using var model = LlmModel.Load(dir, new LlmLoadOptions { Devices = [slot!], ContextLength = 1024 });
        using var vision = model.CreateVisionEncoder();
        var tiles = new List<float[]>();
        if (columns * rows > 1)
        {
            var canvas = Sine(columns * tile, rows * tile);
            for (int tr = 0; tr < rows; tr++)
            {
                for (int tc = 0; tc < columns; tc++)
                {
                    tiles.Add(Crop(canvas, columns * tile, tc * tile, tr * tile, tile));
                }
            }
        }
        else
        {
            columns = rows = 0;
        }

        var image = ViewPreprocessor.Views(vision.Parameters, Sine(size, size), tiles, columns, rows);
        Assert.Equal(r.GetProperty("image_tokens").GetInt32(), image.TokenCount);
        var embeddings = vision.Encode(image);
        var (maxAbs, _, rms) = Env.Compare(embeddings, expected);
        Console.Error.WriteLine($"{name} {reference} ({(gpu ? "Vulkan" : "CPU")}): embeddings max |diff| {maxAbs:E2}, relative rms {rms:E2}");
        Assert.True(maxAbs < (gpu ? 2e-3f : 2e-4f), $"embeddings differ by {maxAbs}");

        // The reference prompt has one <image> per row; the assembly expands a single placeholder.
        var promptIds = r.GetProperty("prompt_ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        int imageId = promptIds[1];
        var collapsed = promptIds.Where((t, i) => t != imageId || promptIds[i - 1] != imageId).ToArray();
        var mm = MultimodalRegistry.Find(model.Definition.Architecture)!.Assemble(new MultimodalContext(model.Definition, t => Single(model, t)), collapsed, [image], 0);
        Assert.Equal(promptIds.Length, mm.Length);
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

    [Theory]
    [InlineData(1000, 1000, 2, 2)]
    [InlineData(1700, 900, 2, 1)]
    [InlineData(600, 2000, 1, 3)]
    public void Tile_grid_matches_the_reference_processor(int width, int height, int columns, int rows)
        => Assert.Equal((columns, rows), ViewPreprocessor.TileGrid(width, height, 640, 2, 9));

    /// <summary>
    /// The global view (resized to fit, padded to a square) and the tiles of the test pattern of hf_ocr_preprocess_reference.py against
    /// transformers' image processor (reference_preprocess_{width}x{height}.bin): the same tile grid, and pixels within two 8-bit levels
    /// (torchvision's and PIL's antialiased bicubic round differently).
    /// </summary>
    [Theory]
    [InlineData(1300, 900)]
    [InlineData(500, 400)]
    public void Views_match_the_reference_processor(int width, int height)
    {
        string dir = Path.Combine(Home, "models/hf/tiny-random-unlimited-ocr");
        string refPath = Path.Combine(dir, $"reference_preprocess_{width}x{height}.bin");
        Assert.SkipUnless(File.Exists(refPath), $"{refPath} not available");
        var bytes = File.ReadAllBytes(refPath);
        int columns = BitConverter.ToInt32(bytes, 0);
        int rows = BitConverter.ToInt32(bytes, 4);
        var expected = new float[(bytes.Length - 8) / 4];
        Buffer.BlockCopy(bytes, 8, expected, 0, bytes.Length - 8);

        var rgb = new byte[3 * width * height];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int o = 3 * (y * width + x);
                rgb[o] = (byte)((7 * x + 13 * y) % 256);
                rgb[o + 1] = (byte)((3 * x + 5 * y + 80) % 256);
                rgb[o + 2] = (byte)(x * y / 64 % 256);
            }
        }

        using var model = LlmModel.Load(dir, new LlmLoadOptions { Devices = [Env.Cpu()], ContextLength = 64 });
        var p = VisionParameters.Of(model.Definition)!;
        var image = new VisionPreprocessor(p).Process(new ImageInput(width, height, rgb));
        bool tiled = columns * rows > 1;
        Assert.Equal(tiled ? (columns, rows) : (0, 0), (image.TileColumns, image.TileRows));
        Assert.Equal(expected.Length, image.Patches.Length);
        // The global view first in both, then the tiles.
        float worst = 0;
        int off = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            float d = Math.Abs(expected[i] - image.Patches[i]);
            worst = Math.Max(worst, d);
            if (d > 1.5f / 127.5f)
            {
                off++;
            }
        }

        Console.Error.WriteLine($"{width}x{height}: max |diff| {worst:E2}, {off} of {expected.Length} values more than one level off");
        Assert.True(worst <= 2.01f / 127.5f, $"pixels differ by {worst * 127.5f} levels");
        Assert.True(off < expected.Length / 1000, $"{off} values differ by more than one level");
    }

    /// <summary>
    /// End to end: the test pattern image and the prompt "&lt;image&gt;\nFree OCR." through the generator (preprocessing, views, layout, prompt
    /// assembly) give transformers' processor + generate output (reference_generate_1300x900.json, hf_ocr_preprocess_reference.py).
    /// </summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Image_prompt_generates_like_transformers(bool gpu)
    {
        string dir = Path.Combine(Home, "models/hf/tiny-random-unlimited-ocr");
        string refPath = Path.Combine(dir, "reference_generate_1300x900.json");
        Assert.SkipUnless(File.Exists(refPath), $"{refPath} not available");
        var slot = gpu ? Env.Vulkan() : Env.Cpu();
        Assert.SkipWhen(slot is null, "no Vulkan device");
        using var doc = JsonDocument.Parse(File.ReadAllBytes(refPath));
        var r = doc.RootElement;
        var expected = r.GetProperty("greedy").EnumerateArray().Select(e => e.GetInt32()).ToArray();
        const int width = 1300;
        const int height = 900;
        var rgb = new byte[3 * width * height];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int o = 3 * (y * width + x);
                rgb[o] = (byte)((7 * x + 13 * y) % 256);
                rgb[o + 1] = (byte)((3 * x + 5 * y + 80) % 256);
                rgb[o + 2] = (byte)(x * y / 64 % 256);
            }
        }

        using var model = LlmModel.Load(dir, new LlmLoadOptions { Devices = [slot!], ContextLength = 1024 });
        using var vision = model.CreateVisionEncoder();
        using var session = model.CreateSession(WellKnown.DecodeMultimodalGraph);
        var ids = new Generator(session, model.Tokenizer, vision)
            .Generate(r.GetProperty("prompt").GetString()!, [new ImageInput(width, height, rgb)], new GenerationOptions { MaxNewTokens = expected.Length, Sampling = SamplingOptions.Greedy })
            .Select(t => t.Id).ToArray();
        Assert.Equal(r.GetProperty("prompt_ids").GetArrayLength(), session.Position - ids.Length + 1);
        Assert.Equal(expected, ids);
    }

    /// <summary>The synthetic normalized image of hf_ocr_image_reference.py, channels first.</summary>
    private static float[] Sine(int width, int height)
    {
        var v = new float[3 * width * height];
        for (int c = 0; c < 3; c++)
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    v[(c * height + y) * width + x] = (float)Math.Sin(0.031 * x + 0.017 * (c + 1) * y + 0.5 * c);
                }
            }
        }

        return v;
    }

    private static float[] Crop(float[] chw, int width, int x0, int y0, int size)
    {
        int height = chw.Length / 3 / width;
        var v = new float[3 * size * size];
        for (int c = 0; c < 3; c++)
        {
            for (int y = 0; y < size; y++)
            {
                Array.Copy(chw, (c * height + y0 + y) * width + x0, v, (c * size + y) * size, size);
            }
        }

        return v;
    }

    private static int? Single(LlmModel model, string text)
    {
        var ids = model.Tokenizer.Encode(text, addSpecial: false, parseSpecial: true);
        return ids.Length == 1 ? ids[0] : null;
    }
}
