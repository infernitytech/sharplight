using Sharplight.Gguf;
using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Families.Tests;

/// <summary>
/// Quantized Qwen3.8-Flash-Next GGUFs (Unsloth UD-IQ1_M, UD-IQ1_S, UD-Q2_K_XL) store the n-gram embedding's small depthwise convolution
/// in BF16 instead of F32. The tiny model with that weight in BF16 builds and derives exactly the dense convolution of the same values
/// stored in F32.
/// </summary>
public class NGramConvPrecisionTests
{
    private static readonly string s_model = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models", "gguf",
        "tiny-random-qwen3.8-flash-next-sparse-f32.gguf");

    static NGramConvPrecisionTests() => BuiltinAdapters.RegisterAll();

    [Theory]
    [InlineData("bf16")]
    [InlineData("f16")]
    public void HalfPrecisionConv_DerivesTheF32Convolution(string type)
    {
        Assert.SkipUnless(File.Exists(s_model), $"{s_model} not present");
        var source = GgufImporter.Open(s_model);
        try
        {
            var convs = source.Weights.Keys.Where(k => k.EndsWith(".ngram.conv1d.weight", StringComparison.Ordinal)).ToList();
            Assert.NotEmpty(convs);
            var half = DataType.Get(type);
            var rounded = new Dictionary<string, WeightInfo>(source.Weights, StringComparer.Ordinal);
            var stored = new Dictionary<string, WeightInfo>(source.Weights, StringComparer.Ordinal);
            foreach (string name in convs)
            {
                var conv = source.Weights[name];
                float[] values = [.. System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(conv.ReadAll())];
                byte[] halves = new byte[values.Length * 2];
                var asFloats = new float[values.Length];
                for (int i = 0; i < values.Length; i++)
                {
                    ushort bits = type == "bf16" ? (ushort)(BitConverter.SingleToUInt32Bits(values[i]) >> 16) : BitConverter.HalfToUInt16Bits((Half)values[i]);
                    BitConverter.TryWriteBytes(halves.AsSpan(i * 2), bits);
                    asFloats[i] = type == "bf16" ? BitConverter.UInt32BitsToSingle((uint)bits << 16) : (float)BitConverter.UInt16BitsToHalf(bits);
                }

                stored[name] = WeightInfo.Derived(name, half, conv.Shape, () => halves);
                rounded[name] = WeightInfo.Derived(name, DataType.F32, conv.Shape, () => [.. asFloats.SelectMany(BitConverter.GetBytes)]);
            }

            var expected = Build(source, rounded);
            var actual = Build(source, stored);
            var dense = expected.Weights.Keys.Where(k => k.EndsWith(".ngram.conv1d.dense", StringComparison.Ordinal)).ToList();
            Assert.Equal(convs.Count, dense.Count);
            foreach (string name in dense)
            {
                Assert.Equal(expected.Weights[name].ReadAll(), actual.Weights[name].ReadAll());
            }
        }
        finally
        {
            foreach (var resource in source.OwnedResources)
            {
                resource.Dispose();
            }
        }
    }

    private static ModelDefinition Build(ModelSource source, Dictionary<string, WeightInfo> weights)
        => ModelAdapterRegistry.Build(new ModelSourceSet
        {
            Primary = new ModelSource
            {
                Format = source.Format, Architecture = source.Architecture, Metadata = source.Metadata, Weights = weights, Path = source.Path,
                FormatData = source.FormatData,
            },
        });
}
