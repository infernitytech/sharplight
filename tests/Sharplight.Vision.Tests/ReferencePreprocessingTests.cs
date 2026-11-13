namespace Sharplight.Vision.Tests;

/// <summary>Developer check against transformers' pixel_values for a photo that needs resizing (needs .work/ref from .work/ref_vision.py).</summary>
public class ReferencePreprocessingTests(ITestOutputHelper output)
{
    [Fact]
    public void Resized_photo_patches_match_transformers()
    {
        string root = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "../../../../../.work/ref"));
        string rgbPath = Path.Combine(root, "test-1.rgb");
        string npy = Path.Combine(root, "test-1_pixels.npy");
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
