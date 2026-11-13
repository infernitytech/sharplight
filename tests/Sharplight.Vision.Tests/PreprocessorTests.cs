using Sharplight.Architectures;

namespace Sharplight.Vision.Tests;

public class PreprocessorTests
{
    /// <summary>Qwen3-VL family geometry: patch 16, merge 2 (factor 32), 48×48 learned positions, official pixel bounds.</summary>
    private static readonly VisionParameters Qwen = new()
    {
        LayerCount = 1, HiddenSize = 64, FfnHiddenSize = 64, HeadCount = 1, PatchSize = 16, TemporalPatchSize = 2, SpatialMergeSize = 2,
        PositionEmbeddingCount = 2304, OutHiddenSize = 64, MinPixels = 65536, MaxPixels = 16777216,
    };

    // Reference values from transformers' Qwen2-VL smart_resize (factor 32, min 65536, max 16777216).
    [Theory]
    [InlineData(640, 488, 640, 480)]
    [InlineData(320, 320, 320, 320)]
    [InlineData(100, 100, 256, 256)]
    [InlineData(8000, 6000, 4704, 3520)]
    [InlineData(3000, 50, 3008, 64)]
    [InlineData(47, 33, 320, 224)]
    [InlineData(1920, 1080, 1920, 1088)]
    [InlineData(333, 500, 320, 512)]
    public void Target_size_matches_the_reference_smart_resize(int w, int h, int tw, int th)
        => Assert.Equal((tw, th), new VisionPreprocessor(Qwen).TargetSize(w, h));

    [Fact]
    public void Token_limits_override_the_model_bounds()
        => Assert.Equal((672, 384), new VisionPreprocessor(Qwen, new ImageTokenLimits(4, 256)).TargetSize(1920, 1080));

    /// <summary>Pixel (x, y) = (x, y, 0): checks normalization, merge-window patch order and the (channel, frame, row, col) patch layout.</summary>
    [Fact]
    public void Patches_are_in_merge_window_order_with_channel_frame_row_col_layout()
    {
        var p = Qwen with { MinPixels = 0, MaxPixels = 0, ImageMean = [0f, 0.5f, 1f], ImageStd = [1f, 0.5f, 2f] };
        int w = 64, h = 96;   // 4 × 6 patches
        var rgb = new byte[w * h * 3];
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++) { int o = (y * w + x) * 3; rgb[o] = (byte)x; rgb[o + 1] = (byte)y; rgb[o + 2] = 255; }
        }

        var im = new VisionPreprocessor(p).Process(new ImageInput(w, h, rgb));
        Assert.Equal((6, 4, 3, 2), (im.GridHeight, im.GridWidth, im.MergedHeight, im.MergedWidth));
        Assert.Equal(24, im.PatchCount);
        Assert.Equal(6, im.TokenCount);
        int K = p.PatchDim; Assert.Equal(1536, K);
        Assert.Equal(24 * K, im.Patches.Length);

        // Window order: patches 0..3 are the first 2×2 block (rows 0-1, cols 0-1), 4..7 the next block to the right.
        Assert.Equal([0, 0, 0, 1, 1, 0, 1, 1, 0, 2, 0, 3, 1, 2, 1, 3], im.Positions[..16]);
        for (int i = 0; i < 24; i++)
        {
            int row = im.Positions[i * 2], col = im.Positions[i * 2 + 1];
            for (int c = 0; c < 3; c++)
            {
                for (int t = 0; t < 2; t++)
                {
                    foreach (var (y, x) in new[] { (0, 0), (3, 5), (15, 15) })
                    {
                        int px = col * 16 + x, py = row * 16 + y;
                        float raw = c == 0 ? px : c == 1 ? py : 255;
                        float expected = (raw / 255f - p.ImageMean[c]) / p.ImageStd[c];
                        float got = im.Patches[i * K + ((c * 2 + t) * 16 + y) * 16 + x];
                        Assert.True(Math.Abs(got - expected) < 1e-5, $"patch {i} c{c} t{t} ({y},{x}): {got} vs {expected}");
                    }
                }
            }
        }
    }

    [Fact]
    public void Position_taps_are_bilinear_with_align_corners()
    {
        var p = Qwen with { MinPixels = 0, MaxPixels = 0 };
        var im = new VisionPreprocessor(p).Process(new ImageInput(64, 96, new byte[64 * 96 * 3]));   // 6 rows × 4 cols
        for (int i = 0; i < im.PatchCount; i++)
        {
            float sum = 0;
            for (int k = 0; k < 4; k++) { sum += im.EmbeddingWeight[i * 4 + k]; Assert.InRange(im.EmbeddingIndex[i * 4 + k], 0, 2303); }
            Assert.Equal(1f, sum, 1e-5f);
        }
        // Patch 0 is (row 0, col 0): all weight on grid cell (0, 0). The window-order patch at (row 5, col 3) is the last grid cell (47, 47).
        Assert.Equal((0, 1f), (im.EmbeddingIndex[0], im.EmbeddingWeight[0]));
        int last = Enumerable.Range(0, im.PatchCount).Single(i => im.Positions[i * 2] == 5 && im.Positions[i * 2 + 1] == 3);
        Assert.Equal(47 * 48 + 47, im.EmbeddingIndex[last * 4]);
        Assert.Equal(1f, im.EmbeddingWeight[last * 4], 1e-5f);
        // (row 1, col 0) of 6 rows → source row 1·47/5 = 9.4: taps rows 9 and 10 with weights 0.6 / 0.4, column 0.
        int r1 = Enumerable.Range(0, im.PatchCount).Single(i => im.Positions[i * 2] == 1 && im.Positions[i * 2 + 1] == 0);
        Assert.Equal([9 * 48, 9 * 48 + 1, 10 * 48, 10 * 48 + 1], im.EmbeddingIndex[(r1 * 4)..(r1 * 4 + 4)]);
        Assert.Equal(0.6f, im.EmbeddingWeight[r1 * 4], 1e-4f);
        Assert.Equal(0.4f, im.EmbeddingWeight[r1 * 4 + 2], 1e-4f);
    }

    [Fact]
    public void Resize_keeps_flat_colours_and_identity_sizes()
    {
        var flat = new byte[50 * 30 * 3];
        for (int i = 0; i < flat.Length; i += 3) { flat[i] = 200; flat[i + 1] = 100; flat[i + 2] = 7; }
        foreach (var (w, h) in new[] { (25, 15), (100, 60), (64, 32), (50, 30) })
        {
            var r = VisionPreprocessor.ResizeBicubic(flat, 50, 30, w, h);
            Assert.Equal(w * h * 3, r.Length);
            for (int i = 0; i < r.Length; i += 3)
            {
                Assert.Equal((200, 100, 7), (r[i], r[i + 1], r[i + 2]));
            }
        }
        var rnd = new byte[50 * 30 * 3]; new Random(1).NextBytes(rnd);
        Assert.Equal(rnd, VisionPreprocessor.ResizeBicubic(rnd, 50, 30, 50, 30));
        // 2× downscale of a 2-pixel checkerboard averages to mid-grey (antialiased filter).
        var chk = new byte[16 * 16 * 3];
        for (int y = 0; y < 16; y++)
        {
            for (int x = 0; x < 16; x++)
            {
                for (int c = 0; c < 3; c++)
                {
                    chk[(y * 16 + x) * 3 + c] = (byte)(((x + y) & 1) * 255);
                }
            }
        }

        var small = VisionPreprocessor.ResizeBicubic(chk, 16, 16, 4, 4);
        Assert.All(small, v => Assert.InRange((int)v, 120, 135));
    }

    [Fact]
    public void Image_validation_and_parameter_round_trip()
    {
        Assert.Throws<ArgumentException>(() => new VisionPreprocessor(Qwen).Process(new ImageInput(10, 10, new byte[5])));
        var props = new Dictionary<string, object>();
        Tiny.Vision.WriteProperties(props);
        var back = VisionParameters.FromProperties(props)!;
        Assert.Equal(Tiny.Vision with { ImageMean = [], ImageStd = [] }, back with { ImageMean = [], ImageStd = [] });
        Assert.Equal(Tiny.Vision.ImageMean, back.ImageMean);
        Assert.Null(VisionParameters.FromProperties(new Dictionary<string, object>()));
    }

    [Fact]
    public void Qwen_prompt_assembly_expands_the_placeholder_with_mrope_positions()
    {
        const int Start = 10, Pad = 11, End = 12;
        var ctx = new MultimodalContext(Tiny.Build(true, false), s => s switch { "<|image_pad|>" => Pad, _ => null });
        var p = Qwen with { MinPixels = 0, MaxPixels = 0 };
        var im = new VisionPreprocessor(p).Process(new ImageInput(96, 64, new byte[96 * 64 * 3]));   // 4 rows × 6 cols of patches → 2 × 3 tokens
        Assert.Equal((2, 3), (im.MergedHeight, im.MergedWidth));
        var arch = MultimodalRegistry.Find("qwen3_5") ?? new Qwen35Multimodal();
        var prompt = arch.Assemble(ctx, [1, 2, Start, Pad, End, 3], [im], startPosition: 5);

        Assert.Equal(3 + 6 + 2, prompt.Length);
        Assert.Equal([1, 2, Start, 0, 0, 0, 0, 0, 0, End, 3], prompt.Tokens);
        Assert.Equal([-1, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1], prompt.ImageIndex);
        Assert.Equal([0, 1, 2, 3, 4, 5], prompt.ImageRow[3..9]);
        // Text before: 5, 6, 7 on all of t/h/w (axis 3 unused). The image starts at 8: t = 8, h = 8 + row, w = 8 + col.
        // The text after continues at 8 + max(3, 2) = 11.
        int Pos(int row, int axis) => prompt.Positions[row * 4 + axis];
        Assert.Equal([5, 5, 5, 0], new[] { Pos(0, 0), Pos(0, 1), Pos(0, 2), Pos(0, 3) });
        Assert.Equal([7, 7, 7, 0], new[] { Pos(2, 0), Pos(2, 1), Pos(2, 2), Pos(2, 3) });
        Assert.Equal([(8, 8, 8), (8, 8, 9), (8, 8, 10), (8, 9, 8), (8, 9, 9), (8, 9, 10)], Enumerable.Range(3, 6).Select(r => (Pos(r, 0), Pos(r, 1), Pos(r, 2))).ToArray());
        Assert.Equal([11, 11, 11, 0], new[] { Pos(9, 0), Pos(9, 1), Pos(9, 2), Pos(9, 3) });
        Assert.Equal(13, prompt.NextPosition);
        Assert.Throws<ArgumentException>(() => arch.Assemble(ctx, [1, 2], [im], 0));
        Assert.Throws<ArgumentException>(() => arch.Assemble(ctx, [1, Pad, Pad], [im], 0));
    }
}

/// <summary>Developer check against transformers' pixel_values for a photo that needs resizing (needs .work/ref from .work/ref_vision.py).</summary>
public class ReferencePreprocessingTests(ITestOutputHelper output)
{
    [Fact]
    public void Resized_photo_patches_match_transformers()
    {
        string root = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../../.work/ref"));
        string rgbPath = Path.Combine(root, "test-1.rgb"), npy = Path.Combine(root, "test-1_pixels.npy");
        Assert.SkipUnless(File.Exists(rgbPath) && File.Exists(npy), "reference pixel_values not generated (.work/ref)");
        var raw = File.ReadAllBytes(npy);
        int header = 10 + BitConverter.ToUInt16(raw, 8);
        var expected = new float[(raw.Length - header) / 4];
        Buffer.BlockCopy(raw, header, expected, 0, raw.Length - header);

        var p = new VisionParameters
        {
            LayerCount = 1, HiddenSize = 64, FfnHiddenSize = 64, HeadCount = 1, PatchSize = 16, TemporalPatchSize = 2, SpatialMergeSize = 2,
            PositionEmbeddingCount = 2304, OutHiddenSize = 64, MinPixels = 65536, MaxPixels = 16777216,
        };
        var im = new VisionPreprocessor(p).Process(new ImageInput(640, 488, File.ReadAllBytes(rgbPath)));
        Assert.Equal(expected.Length, im.Patches.Length);
        var (abs, rel, rms) = Env.Compare(im.Patches, expected);
        // Values are in [-1, 1] steps of 2/255; our filter vs torch's antialiased bicubic differ by at most a rounding step.
        long off = 0; for (int i = 0; i < expected.Length; i++)
        {
            if (Math.Abs(im.Patches[i] - expected[i]) > 2.5f / 255)
            {
                off++;
            }
        }

        Console.Error.WriteLine($"preprocessing vs transformers (640x488 -> {im.Width}x{im.Height}): max abs {abs:E3}, relative RMS {rms:E3}, {off}/{expected.Length} values off by more than one 8-bit step");
        output.WriteLine($"max abs {abs}");
        Assert.True(abs < 0.05f);
    }
}
