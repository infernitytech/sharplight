using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;
using Xunit;

namespace Sharplight.Package.Tests;

/// <summary><see cref="WeightInfo.ScaleRowBlock"/>: kept by the aligned views, stored in packages (the WLAY section).</summary>
public class WeightLayoutTests
{
    private static WeightInfo Fp8(string name, long rows, int block)
        => new()
        {
            Name = name, Type = DataType.F8E4M3_B128, Shape = new Shape(rows, 256),
            Source = new BytesSource(new byte[DataType.F8E4M3_B128.ByteSize(rows * 256)]), ScaleRowBlock = block,
        };

    /// <summary>Row slices and concatenations keep the block only where the groups stay aligned; reshapes where rows stay whole groups.</summary>
    [Fact]
    public void ViewsKeepAlignedScaleRowBlocks()
    {
        var w = Fp8("w", 512, 128);
        Assert.Equal(128, w.RowSlice("a", 128, 256).ScaleRowBlock);
        Assert.Equal(0, w.RowSlice("b", 64, 256).ScaleRowBlock);
        Assert.Equal(128, WeightInfo.ConcatRows("c", [w, Fp8("v", 256, 128)]).ScaleRowBlock);
        Assert.Equal(0, WeightInfo.ConcatRows("d", [w, Fp8("v", 256, 0)]).ScaleRowBlock);
        Assert.Equal(0, WeightInfo.ConcatRows("e", [w.RowSlice("s", 0, 192), w]).ScaleRowBlock);
        Assert.Equal(128, WeightInfo.Stack("f", [w, w]).ScaleRowBlock);
        Assert.Equal(128, w.As("g", new Shape(2, 256, 256)).ScaleRowBlock);
        Assert.Equal(0, w.As("h", new Shape(8, 64, 256)).ScaleRowBlock);
    }

    /// <summary>A package stores the block of the weights that have one; the others read back without.</summary>
    [Fact]
    public void ScaleRowBlockRoundTrips()
    {
        using var tmp = new TempDir();
        var weights = new Dictionary<string, WeightInfo> { ["w"] = Fp8("w", 256, 128), ["v"] = Fp8("v", 256, 0) };
        var def = new ModelDefinition { Architecture = "test_arch", Name = "fp8", Weights = weights, Graphs = new Dictionary<string, ComputeGraph>() };
        var p = tmp.File("fp8.slm");
        ModelPackage.Save(p, def, []);
        using var f = new PackageFile(p);
        Assert.Equal(128, f.Weights["w"].ScaleRowBlock);
        Assert.Equal(0, f.Weights["v"].ScaleRowBlock);
        Assert.Contains(ModelPackageReader.Open(p).Sections, s => s.Kind == "WLAY");
    }
}
