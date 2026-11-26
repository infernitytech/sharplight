using System.Text.Json;
using Sharplight.Llm;

namespace Sharplight.Vision.Tests;

/// <summary>
/// Image preprocessing of the Gemma 3 (fixed square, bilinear) and Idefics3 / SmolVLM (longest-edge and tile-multiple Lanczos resizes, tile grid
/// and overview) towers against transformers' PIL image processors on synthetic 8-bit images (reference_preprocess.json next to the tiny checkpoints of
/// mk_tiny_siglip_vl.py, written by siglip_vl_preprocess_reference.py; the tower inputs come from the converted mmproj files).
/// </summary>
public class SiglipVlPreprocessTests
{
    private static readonly string Home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

    [Theory]
    [InlineData("tiny-random-gemma3-vl")]
    [InlineData("tiny-random-idefics3")]
    public void Preprocessing_matches_transformers(string name)
    {
        string refPath = Path.Combine(Home, "models/hf", name, "reference_preprocess.json");
        string gguf = Path.Combine(Home, "models/gguf", $"{name}-f32.gguf");
        string mmproj = Path.Combine(Home, "models/gguf", $"mmproj-{name}-f32.gguf");
        Assert.SkipUnless(File.Exists(refPath) && File.Exists(gguf) && File.Exists(mmproj), $"{name}: reference_preprocess.json, GGUF or mmproj not available");
        using var model = LlmModel.Load(TestHost.Loader, gguf, new LlmLoadOptions { Devices = [Env.Cpu()], ContextLength = 64, Auxiliary = { [AuxiliaryRoles.Mmproj] = mmproj } });
        using var vision = model.CreateVisionEncoder();
        var pre = vision.Preprocessor;
        using var doc = JsonDocument.Parse(File.ReadAllBytes(refPath));
        foreach (var c in doc.RootElement.GetProperty("cases").EnumerateArray())
        {
            int width = c.GetProperty("width").GetInt32();
            int height = c.GetProperty("height").GetInt32();
            var rgb = new byte[3 * width * height];
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    for (int ch = 0; ch < 3; ch++)
                    {
                        rgb[(y * width + x) * 3 + ch] = (byte)((7 * x + 3 * y + 50 * ch + x * y % 31) % 256);
                    }
                }
            }

            var ours = pre.Process(new ImageInput(width, height, rgb));
            var pixels = c.GetProperty("pixels").EnumerateArray().Select(e => e.GetSingle()).ToArray();
            int side = pre.Parameters.TileSize > 0 ? pre.Parameters.TileSize : pre.Parameters.ImageSize;
            int plane = side * side;
            var views = Enumerable.Range(0, pixels.Length / (3 * plane)).Select(v => Interleave(pixels, v * 3 * plane, side)).ToList();
            PreprocessedImage expected;
            if (c.TryGetProperty("rows", out var rows))
            {
                int columns = c.GetProperty("cols").GetInt32();
                Assert.Equal(columns, ours.TileColumns);
                Assert.Equal(rows.GetInt32(), ours.TileRows);
                expected = pre.PatchifyTiles(views[^1], views.Take(views.Count - 1).ToList(), columns, rows.GetInt32());
            }
            else
            {
                expected = pre.Patchify(Assert.Single(views), side, side);
            }

            var (maxAbs, _, rms) = Env.Compare(ours.Patches, expected.Patches);
            Console.Error.WriteLine($"{name} {width}x{height}: {views.Count} view(s), max |diff| {maxAbs:E2}, rms {rms:E2}");
            // One 8-bit step is 2/255 after the (x/255 - 0.5)/0.5 normalization. Both passes round to 8 bits like Pillow, but Pillow 13 runs the
            // vertical pass first for some sizes (older Pillow always horizontal first, as here), which moves many pixels by one step and a few
            // by more next to the Lanczos kernel's overshoot.
            Assert.True(maxAbs <= 12f * 2 / 255f, $"{name} {width}x{height}: pixels differ by {maxAbs}");
            Assert.True(rms < 1.5e-2, $"{name} {width}x{height}: pixel rms {rms}");
        }
    }

    // One channels-first view [3, side, side] at offset as interleaved RGB.
    private static float[] Interleave(float[] chw, int offset, int side)
    {
        var v = new float[3 * side * side];
        for (int c = 0; c < 3; c++)
        {
            for (int i = 0; i < side * side; i++)
            {
                v[i * 3 + c] = chw[offset + c * side * side + i];
            }
        }

        return v;
    }
}
