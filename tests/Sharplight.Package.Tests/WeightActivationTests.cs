using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;
using Xunit;

namespace Sharplight.Package.Tests;

/// <summary><see cref="WeightInfo.Activations"/>: kept by every view, stored in packages (the WACT section) next to the layout facts.</summary>
public class WeightActivationTests
{
    private static WeightInfo Nvfp4(string name, ActivationScheme scheme, int scaleRowBlock = 0)
        => new()
        {
            Name = name, Type = DataType.NVFP4, Shape = new Shape(256, 256), Source = new BytesSource(new byte[DataType.NVFP4.ByteSize(256 * 256)]),
            Activations = scheme, ScaleRowBlock = scaleRowBlock,
        };

    /// <summary>Views keep the scheme; a tensor joined from parts is weight-only when one part is, quantized when all are.</summary>
    [Fact]
    public void ViewsKeepTheScheme()
    {
        var w = Nvfp4("w", ActivationScheme.Unquantized);
        var q = Nvfp4("q", ActivationScheme.Quantized);
        Assert.Equal(ActivationScheme.Unquantized, w.RowSlice("a", 0, 128).Activations);
        Assert.Equal(ActivationScheme.Unquantized, w.As("b", new Shape(2, 128, 256)).Activations);
        Assert.Equal(ActivationScheme.Unquantized, WeightInfo.Stack("c", [w, q]).Activations);
        Assert.Equal(ActivationScheme.Quantized, WeightInfo.ConcatRows("d", [q, q]).Activations);
        Assert.Equal(ActivationScheme.Unspecified, WeightInfo.ConcatRows("e", [q, Nvfp4("u", ActivationScheme.Unspecified)]).Activations);
    }

    /// <summary>A package stores the scheme of the weights that have one, and the layout section does not drop it.</summary>
    [Fact]
    public void SchemeRoundTrips()
    {
        using var tmp = new TempDir();
        var weights = new Dictionary<string, WeightInfo>
        {
            ["w"] = Nvfp4("w", ActivationScheme.Unquantized, scaleRowBlock: 128),
            ["q"] = Nvfp4("q", ActivationScheme.Quantized),
            ["u"] = Nvfp4("u", ActivationScheme.Unspecified),
        };
        var def = new ModelDefinition { Architecture = "test_arch", Name = "nvfp4", Weights = weights, Graphs = new Dictionary<string, ComputeGraph>() };
        var p = tmp.File("nvfp4.slm");
        ModelPackage.Save(p, def, []);
        using var f = new PackageFile(p);
        Assert.Equal(ActivationScheme.Unquantized, f.Weights["w"].Activations);
        Assert.Equal(128, f.Weights["w"].ScaleRowBlock);
        Assert.Equal(ActivationScheme.Quantized, f.Weights["q"].Activations);
        Assert.Equal(ActivationScheme.Unspecified, f.Weights["u"].Activations);
        Assert.Contains(ModelPackageReader.Open(p).Sections, s => s.Kind == "WACT");
    }
}
