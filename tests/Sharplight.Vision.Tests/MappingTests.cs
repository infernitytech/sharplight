using Sharplight.Gguf;
using Sharplight.HuggingFace;
using Sharplight.Tensors;

namespace Sharplight.Vision.Tests;

/// <summary>The Hugging Face importer and the GGUF mmproj importer must produce the same canonical vision description.</summary>
public class MappingTests(ITestOutputHelper output)
{
    private static bool Have(string p) => File.Exists(p) || Directory.Exists(p);

    [Fact]
    public void Hf_and_mmproj_give_the_same_canonical_tower()
    {
        Assert.SkipUnless(Have(Env.Hf08) && Have(Env.Mmproj08), "Qwen3.5-0.8B HF checkpoint or its converted mmproj (.work/mmproj-0.8B-f32.gguf) not available");
        using var _ = new Disposer(out var res);
        var hf = HuggingFaceImporter.Open(Env.Hf08);
        var gg = GgufImporter.Open(Env.Mmproj08);
        res.AddRange(hf.OwnedResources);
        res.AddRange(gg.OwnedResources);

        Assert.Equal("qwen3_5", gg.Architecture);
        var ph = VisionParameters.FromMetadata(hf.Metadata, "qwen3_5.vision.")!;
        var pg = VisionParameters.FromMetadata(gg.Metadata, "qwen3_5.vision.")!;
        Assert.Equal(ph with { ImageMean = [], ImageStd = [] }, pg with { ImageMean = [], ImageStd = [] });
        Assert.Equal(ph.ImageMean, pg.ImageMean);
        Assert.Equal(ph.ImageStd, pg.ImageStd);

        var hw = hf.Weights.Where(k => k.Key.StartsWith("vision.")).ToDictionary(k => k.Key, k => k.Value);
        var gw = gg.Weights.ToDictionary(k => k.Key, k => k.Value);
        Assert.Equal(hw.Keys.Order(), gw.Keys.Order());
        double worst = 0;
        foreach (var (name, h) in hw)
        {
            var g = gw[name];
            Assert.Equal(h.Shape.ToString(), g.Shape.ToString());
            var a = HfTransforms.DecodeF32(h);
            var b = HfTransforms.DecodeF32(g);
            double maxDiff = 0;
            for (int i = 0; i < a.Length; i++)
            {
                maxDiff = Math.Max(maxDiff, Math.Abs(a[i] - b[i]));
            }

            worst = Math.Max(worst, maxDiff);
            Assert.True(maxDiff == 0, $"{name}: max |hf - gguf| = {maxDiff}");
        }
        Console.Error.WriteLine($"HF vs mmproj: {hw.Count} canonical vision weights identical (max diff {worst}), hyper-parameters equal");
    }

    [Fact]
    public void Mmproj_27b_has_the_canonical_tower_of_a_qwen3_vit()
    {
        Assert.SkipUnless(Have(Env.Mmproj27), "27B mmproj not available");
        var src = GgufImporter.Open(Env.Mmproj27);
        using var _ = new Disposer(src.OwnedResources);
        var p = VisionParameters.FromMetadata(src.Metadata, "qwen3_5.vision.")!;
        Assert.Equal((27, 1152, 4304, 16, 16, 2, 2, 2304, 5120), (p.LayerCount, p.HiddenSize, p.FfnHiddenSize, p.HeadCount, p.PatchSize, p.TemporalPatchSize, p.SpatialMergeSize, p.PositionEmbeddingCount, p.OutHiddenSize));
        Assert.Equal(72, p.HeadDim);
        Assert.Equal(new Shape(1152, 1536), src.Weights["vision.patch_embed.weight"].Shape);
        Assert.Equal(new Shape(3456, 1152), src.Weights["vision.layers.26.attention.qkv.weight"].Shape);
        Assert.Equal(new Shape(5120, 4608), src.Weights["vision.merger.down.weight"].Shape);
        Assert.Equal(27 * 12 + 9, src.Weights.Count);
    }
}
