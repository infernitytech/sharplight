using Sharplight.Runtime;
using Xunit;

namespace Sharplight.Import.Onnx.Tests;

/// <summary>StringSplit (substrings in the ONNX layout through <see cref="OnnxOutputLayout"/>) and the com.microsoft WordConvEmbedding.</summary>
public sealed class TextOpTests
{
    [Fact]
    public void StringSplit_PadsTheSubstringsOfEveryString()
    {
        var model = new TestModel().Opset(20).Input(TestModel.ValueInfo("x", 8, 2, 2)).Output(TestModel.ValueInfo("y", 8, 2, 2, "k"))
            .Output(TestModel.ValueInfo("z", 7, 2, 2)).Node("StringSplit", ["x"], ["y", "z"], TestModel.AttrS("delimiter", "-"));
        var definition = model.Build();
        var graph = definition.Graph(OnnxGraphAdapter.MainGraph);
        var layout = ((IReadOnlyDictionary<string, OnnxOutputLayout>)definition.Properties[OnnxGraphAdapter.OutputLayoutsProperty])["y"];
        using var compiled = CompiledModel.Compile(graph, new PlacementOptions { Devices = [new DeviceSlot { Device = Providers.Cpu }], Compile = new CompileOptions() });
        var run = new RunParameters();
        compiled.SetInput(graph.Inputs[0], ["a-b-c", "d", "", "e-f"]);
        compiled.Run(run);

        var shape = compiled.OutputShape("y", run);
        var tokens = new string[shape.Aggregate(1L, (a, b) => a * b)];
        compiled.GetOutput("y", tokens.AsSpan());
        Assert.Equal([2L, 2, 3], layout.OnnxShape(shape));
        Assert.Equal(["a", "b", "c", "d", "", "", "", "", "", "e", "f", ""], layout.ToOnnx<string>(tokens, shape));
        var counts = new int[4];
        compiled.GetOutput<int>("z", counts);
        Assert.Equal([3, 1, 1, 2], counts);
    }

    [Fact]
    public void WordConvEmbedding_MatchesTheReference()
    {
        const int Filters = 4;
        const int Width = 2;
        const int CharEmbedding = 3;
        const int Vocabulary = 10;
        int[] ids = [3, 0, 3, 5, 0, 9, 7, 4, 2, 1, 8, 9, 1, 8, 0, 0, 0, 0];
        var weight = Rnd(Filters * Width * CharEmbedding, 1);
        var bias = Rnd(Filters, 2);
        var table = Rnd(Vocabulary * CharEmbedding, 3);
        var model = new TestModel().Opset(1, "com.microsoft").Input("seq", 3, 6).Output("y", 3, Filters)
            .InitF32("w", [Filters, 1, Width, CharEmbedding], weight).InitF32("b", [Filters], bias).InitF32("c", [Vocabulary, CharEmbedding], table)
            .ContribNode("WordConvEmbedding", ["seq", "w", "b", "c"], ["y"]);
        string path = Path.Combine(Path.GetTempPath(), $"sl-word-conv-{Guid.NewGuid():N}.onnx");
        File.WriteAllBytes(path, model.ToBytes());
        try
        {
            // The int32 ids travel as an i32 graph input.
            var actual = OnnxExec.Run(path, new OnnxImportOptions(), Providers.Cpu, 1, [[.. ids.Select(v => (float)v)]]).Outputs[0];
            var expected = Reference(ids, 3, 6, weight, bias, table, Filters, Width, CharEmbedding);
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.True(Math.Abs(expected[i] - actual[i]) < 1e-5, $"[{i}] {actual[i]} vs {expected[i]}");
            }
        }
        finally
        {
            File.Delete(path);
        }
    }

    /// <summary>onnxruntime's algorithm: the word length counts the non-zero ids (0 when the first one is padding).</summary>
    private static float[] Reference(int[] ids, int words, int characters, float[] weight, float[] bias, float[] table, int filters, int width, int embedding)
    {
        var result = new float[words * filters];
        for (int w = 0; w < words; w++)
        {
            int length = ids[w * characters] > 0 ? ids.Skip(w * characters).Take(characters).Count(id => id > 0) : 0;
            if (length == 0)
            {
                continue;
            }

            int positions = Math.Max(length, width) - width + 1;
            for (int f = 0; f < filters; f++)
            {
                float best = float.NegativeInfinity;
                for (int t = 0; t < positions; t++)
                {
                    float sum = bias[f];
                    for (int k = 0; k < width; k++)
                    {
                        int id = ids[w * characters + t + k];
                        for (int e = 0; e < embedding; e++)
                        {
                            sum += weight[(f * width + k) * embedding + e] * table[id * embedding + e];
                        }
                    }

                    best = Math.Max(best, MathF.Tanh(sum));
                }

                result[w * filters + f] = best;
            }
        }

        return result;
    }

    private static float[] Rnd(int n, int seed)
    {
        var random = new Random(seed);
        return [.. Enumerable.Range(0, n).Select(_ => (float)(random.NextDouble() - 0.5))];
    }
}
