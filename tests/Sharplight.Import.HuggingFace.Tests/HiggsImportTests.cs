using System.Runtime.InteropServices;
using Xunit;

namespace Sharplight.Import.HuggingFace.Tests;

/// <summary>
/// HIGGS (FLUTE-packed 4-bit vector quantization with a Hadamard rotation), galqiwi/Qwen3-1.7B-HIGGS-4bit: the dense F16 weights the
/// importer produces must equal tools/higgs-reference-dense.py (inverse of FLUTE's packer, float32 dequantization and rotation, rounded
/// to F16) up to one F16 ulp or 1e-6 absolute (the summation order of the rotation differs: butterflies here, a matrix product there).
/// Layers 0 and 27. Skips when the checkpoint is absent.
/// </summary>
public sealed class HiggsImportTests
{
    private static readonly string Dir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models/hf/quant/higgs/Qwen3-1.7B-HIGGS-4bit");

    [Fact]
    public void RotatedDequantizedWeightsMatchTheReference()
    {
        string reference = Path.Combine(Dir, "sharplight-reference", "dense.safetensors");
        if (!File.Exists(reference))
        {
            Assert.Skip("checkpoint not present");
        }

        using var config = HfConfig.Load(Dir);
        using var checkpoint = SafetensorsCheckpoint.Open(Dir);
        using var expected = SafetensorsFile.Open(reference);
        var resolved = HfHiggsWeights.Resolve(config, checkpoint.Tensors);
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
            var x = MemoryMarshal.Cast<byte, ushort>(a);
            var y = MemoryMarshal.Cast<byte, ushort>(b);
            int different = 0;
            for (int i = 0; i < x.Length; i++)
            {
                float fx = (float)BitConverter.UInt16BitsToHalf(x[i]);
                float fy = (float)BitConverter.UInt16BitsToHalf(y[i]);
                bool oneUlp = Math.Abs(x[i] - y[i]) <= 1 && (x[i] ^ y[i]) < 0x8000;
                Assert.True(oneUlp || Math.Abs(fx - fy) < 1e-6f, $"{weightName}[{i}]: {fx} vs {fy}");
                different += x[i] == y[i] ? 0 : 1;
            }

            Assert.True(different < x.Length / 50, $"{weightName}: {different} of {x.Length} differ");
        }

        Assert.DoesNotContain(resolved.Keys, key => key.EndsWith(".tables2", StringComparison.Ordinal));
    }
}
