using Xunit;

namespace Sharplight.Import.HuggingFace.Tests;

/// <summary>
/// VPTQ checkpoints (Tiberiw TinyLlama-1.1B, vector length 8, packed indices, column permutation and normalization; 65536 centroids with
/// a 256-centroid residual, and 4096 centroids without): the dense BF16 weights the importer produces must equal
/// tools/vptq-reference-dense.py (VPTQ's PyTorch <c>dequant</c>) bit for bit on layers 0 and 21. Skips when the checkpoints are absent.
/// </summary>
public sealed class VptqImportTests
{
    private static readonly string Quant = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models/hf/quant/vptq");

    [Theory]
    [InlineData("TinyLlama-1.1B-VPTQ-v8-k65536-256")]
    [InlineData("TinyLlama-1.1B-VPTQ-v8-k4096-0")]
    public void DequantizedWeightsEqualTheReference(string name)
    {
        string dir = Path.Combine(Quant, name);
        string reference = Path.Combine(dir, "sharplight-reference", "dense.safetensors");
        if (!File.Exists(reference))
        {
            Assert.Skip("checkpoint not present");
        }

        using var config = HfConfig.Load(dir);
        using var checkpoint = SafetensorsCheckpoint.Open(dir);
        using var expected = SafetensorsFile.Open(reference);
        var resolved = HfVptqWeights.Resolve(config, checkpoint.Tensors);
        Assert.Equal(14, expected.Tensors.Count);
        foreach (var (weightName, want) in expected.Tensors)
        {
            var got = resolved[weightName].ToWeightInfo();
            var wantInfo = want.ToWeightInfo();
            Assert.Equal(wantInfo.Shape.ToString(), got.Shape.ToString());
            Assert.Equal(wantInfo.Type, got.Type);
            var a = new byte[wantInfo.Source.ByteLength];
            var b = new byte[got.Source.ByteLength];
            wantInfo.Source.Read(0, a);
            got.Source.Read(0, b);
            int first = a.AsSpan().CommonPrefixLength(b);
            Assert.True(first == a.Length, $"{weightName}: first difference at byte {first}");
        }

        Assert.DoesNotContain(resolved.Keys, key => key.EndsWith(".indices", StringComparison.Ordinal));
    }
}
