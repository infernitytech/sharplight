using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;
using Xunit;

namespace Sharplight.Package.Tests;

/// <summary>A package writer holds one derived weight at a time: each is released after it is written (streaming conversion).</summary>
public class DerivedWeightReleaseTests
{
    [Fact]
    public void ReleasedDerivedWeight_IsProducedAgainOnTheNextRead()
    {
        int produced = 0;
        var weight = WeightInfo.Derived("w", DataType.F32, new Shape(2, 2), () =>
        {
            produced++;
            return new byte[16];
        });
        var buffer = new byte[16];

        weight.Source.Read(0, buffer);
        weight.Source.Read(0, buffer);
        Assert.Equal(1, produced);
        weight.Source.Release();
        weight.Source.Read(0, buffer);
        Assert.Equal(2, produced);
    }

    [Fact]
    public void Save_ReleasesDerivedWeights_AndWritesTheirBytes()
    {
        int produced = 0;
        var weight = WeightInfo.Derived("w", DataType.F32, new Shape(4, 8), () =>
        {
            produced++;
            return [.. Enumerable.Range(0, 4 * 8 * 4).Select(i => (byte)i)];
        });
        var stacked = new WeightInfo { Name = "s", Type = DataType.F32, Shape = new Shape(2, 4, 8), Source = new StackedWeightSource([weight.Source, weight.Source]) };
        var weights = new Dictionary<string, WeightInfo> { ["w"] = weight, ["s"] = stacked };
        var b = new GraphBuilder("decode", n => weights.GetValueOrDefault(n));
        var x = b.Input("x", InputRole.Embeddings, DataType.F32, new Shape(Dim.Sym("T"), 8));
        b.Output("y", b.MatMul(x, b.Weight("w"), "proj"));
        var definition = new ModelDefinition
        {
            Architecture = "test_arch",
            Name = "derived",
            Weights = weights,
            Graphs = new Dictionary<string, ComputeGraph> { ["decode"] = b.Build() },
        };
        var path = Path.Combine(Path.GetTempPath(), $"derived-{Guid.NewGuid():N}.slm");
        try
        {
            ModelPackage.Save(path, definition, []);
            int afterSave = produced;
            weight.Source.Read(0, new byte[4]);
            Assert.Equal(afterSave + 1, produced);

            using var file = new PackageFile(path);
            foreach (var (name, size) in new[] { ("w", 128), ("s", 256) })
            {
                Assert.Equal(Enumerable.Range(0, size).Select(i => (byte)(i % 128)), file.Definition.Weights[name].ReadAll());
            }
        }
        finally
        {
            File.Delete(path);
        }
    }
}
