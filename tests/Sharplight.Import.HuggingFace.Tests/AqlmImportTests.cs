using Sharplight.Tensors;
using Xunit;

namespace Sharplight.Import.HuggingFace.Tests;

/// <summary>
/// AQLM checkpoints: every dequantized weight must equal, bit for bit, the reference implementation's (aqlm 1.1.7
/// <c>_dequantize_weight</c> on CPU, tools/mk-tiny-aqlm.py and tools/aqlm-reference-dense.py). Synthetic variants cover several
/// codebooks, out groups larger than one, odd code widths, int8 sign wrap and a BF16 codebook (one codebook only: PyTorch's CPU embedding_bag rounds a BF16 sum of several
/// codebooks half away from zero, this importer round-to-nearest-even, a one-ulp difference on ties); the real checkpoint is
/// TinyLlama-1.1B-Chat AQLM 2-bit 1x16 (layers 0 and 21). Skips when the checkpoints are absent.
/// </summary>
public sealed class AqlmImportTests
{
    private static readonly string Quant = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models/hf/quant");

    public static TheoryData<string> Checkpoints() =>
    [
        "aqlm-synth/aqlm-1x8-g8",
        "aqlm-synth/aqlm-2x8-g8",
        "aqlm-synth/aqlm-4x6-og2",
        "aqlm-synth/aqlm-8x8-g8",
        "aqlm-synth/aqlm-1x10-bf16",
        "aqlm/TinyLlama-1.1B-AQLM-2Bit-1x16",
    ];

    [Theory]
    [MemberData(nameof(Checkpoints))]
    public void DequantizedWeightsAreBitIdenticalToTheReference(string name)
    {
        string dir = Path.Combine(Quant, name);
        string reference = Path.Combine(dir, "sharplight-reference", "dense.safetensors");
        if (!File.Exists(reference))
        {
            return;
        }

        using var checkpoint = SafetensorsCheckpoint.Open(dir);
        using var expected = SafetensorsFile.Open(reference);
        var resolved = HfAqlmWeights.Resolve(checkpoint.Tensors);
        Assert.NotEmpty(expected.Tensors);
        foreach (var (weightName, want) in expected.Tensors)
        {
            var got = resolved[weightName].ToWeightInfo();
            Assert.Equal(want.Type, got.Type);
            var wantBytes = want.ToWeightInfo();
            Assert.Equal(wantBytes.Shape.ToString(), got.Shape.ToString());
            var a = new byte[wantBytes.Source.ByteLength];
            var b = new byte[got.Source.ByteLength];
            wantBytes.Source.Read(0, a);
            got.Source.Read(0, b);
            Assert.True(a.AsSpan().SequenceEqual(b), weightName);
        }

        Assert.DoesNotContain(resolved.Keys, key => key.EndsWith(".codes", StringComparison.Ordinal));
    }

    [Fact]
    public void ImportedAqlmModelHasDenseLinearWeights()
    {
        string dir = Path.Combine(Quant, "aqlm/TinyLlama-1.1B-AQLM-2Bit-1x16");
        if (!Directory.Exists(dir))
        {
            return;
        }

        var source = HuggingFaceImporter.Open(dir);
        try
        {
            Assert.All(source.Weights.Where(w => w.Key.Contains(".attention.", StringComparison.Ordinal) && w.Key.EndsWith(".weight", StringComparison.Ordinal)),
                w => Assert.Equal(DataType.F16, w.Value.Type));
        }
        finally
        {
            foreach (var resource in source.OwnedResources)
            {
                resource.Dispose();
            }
        }
    }
}
