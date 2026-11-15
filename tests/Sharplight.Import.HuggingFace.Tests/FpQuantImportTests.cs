using System.Runtime.InteropServices;
using Xunit;

namespace Sharplight.Import.HuggingFace.Tests;

/// <summary>
/// FP-Quant (MXFP4 / NVFP4 with Hadamard rotation) checkpoints from ISTA-DASLab, Qwen3-1.7B RTN: the dense BF16 weights the importer
/// produces must equal tools/fpquant-reference-dense.py (float32 dequantization and rotation, rounded to BF16) up to one BF16 ulp or
/// 1e-6 absolute (the float32 summation order of the rotation differs). Layers 0 and 27. Skips when the checkpoints are absent.
/// </summary>
public sealed class FpQuantImportTests
{
    private static readonly string Quant = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models/hf/quant/fpquant");

    [Theory]
    [InlineData("Qwen3-1.7B-FPQuant-RTN-MXFP4")]
    [InlineData("Qwen3-1.7B-FPQuant-RTN-NVFP4")]
    public void RotatedDequantizedWeightsMatchTheReference(string name)
    {
        string dir = Path.Combine(Quant, name);
        string reference = Path.Combine(dir, "sharplight-reference", "dense.safetensors");
        if (!File.Exists(reference))
        {
            return;
        }

        using var config = HfConfig.Load(dir);
        using var checkpoint = SafetensorsCheckpoint.Open(dir);
        using var expected = SafetensorsFile.Open(reference);
        var resolved = HfFpQuantWeights.Resolve(config, checkpoint.Tensors);
        Assert.NotEmpty(expected.Tensors);
        foreach (var (weightName, want) in expected.Tensors)
        {
            var got = resolved[weightName].ToWeightInfo();
            var wantInfo = want.ToWeightInfo();
            Assert.Equal(wantInfo.Shape.ToString(), got.Shape.ToString());
            var a = new byte[wantInfo.Source.ByteLength];
            var b = new byte[got.Source.ByteLength];
            wantInfo.Source.Read(0, a);
            got.Source.Read(0, b);
            var x = MemoryMarshal.Cast<byte, ushort>(a);
            var y = MemoryMarshal.Cast<byte, ushort>(b);
            int different = 0;
            for (int i = 0; i < x.Length; i++)
            {
                float fx = BitConverter.Int32BitsToSingle(x[i] << 16);
                float fy = BitConverter.Int32BitsToSingle(y[i] << 16);
                bool oneUlp = Math.Abs(x[i] - y[i]) <= 1 && (x[i] ^ y[i]) < 0x8000;
                // Exact cancellations in the rotation leave float32 rounding noise (|v| < 1e-6) whose sign depends on the summation order.
                Assert.True(oneUlp || Math.Abs(fx - fy) < 1e-6f, $"{weightName}[{i}]: {fx} vs {fy}");
                different += x[i] == y[i] ? 0 : 1;
            }

            Assert.True(different < x.Length / 50, $"{weightName}: {different} of {x.Length} differ");
        }

        Assert.DoesNotContain(resolved.Keys, key => key.EndsWith(".qweight", StringComparison.Ordinal));
    }
}
