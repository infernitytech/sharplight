using Xunit;

namespace Sharplight.Import.HuggingFace.Tests;

/// <summary>
/// ExLlamaV2 checkpoints (turboderp TinyLlama-1B EXL2 at 2.5 and 4.0 bits per weight: groups of 2 to 8 bits, row permutation): the
/// dense F16 weights the importer produces must equal exllamav2's own <c>reconstruct</c> (tools/exl2-reference-dense.py) bit for bit on
/// layers 0 and 21 and the quantized lm_head. Skips when the checkpoints are absent.
/// </summary>
public sealed class Exl2ImportTests
{
    private static readonly string Quant = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models/hf/quant/exl2");

    [Theory]
    [InlineData("TinyLlama-1B-exl2-4.0bpw")]
    [InlineData("TinyLlama-1B-exl2-2.5bpw")]
    public void DequantizedWeightsEqualTheReference(string name)
    {
        string dir = Path.Combine(Quant, name);
        string reference = Path.Combine(dir, "sharplight-reference", "dense.safetensors");
        if (!File.Exists(reference))
        {
            Assert.Skip("checkpoint not present");
        }

        using var checkpoint = SafetensorsCheckpoint.Open(dir);
        using var expected = SafetensorsFile.Open(reference);
        var resolved = HfExl2Weights.Resolve(checkpoint.Tensors);
        Assert.Equal(15, expected.Tensors.Count);
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

        Assert.DoesNotContain(resolved.Keys, key => key.EndsWith(".q_weight", StringComparison.Ordinal));
    }
}
