using Sharplight.Graph;
using Sharplight.Graph.Optimization;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tensors;
using Xunit;

namespace Sharplight.Package.Tests;

public class PackageOptimizerTests
{
    /// <summary>A definition whose graph the default pipeline shrinks: a common subexpression and a dead node.</summary>
    private static ModelDefinition RedundantDefinition()
    {
        var b = new GraphBuilder("decode", _ => null);
        var x = b.Input("x", InputRole.Embeddings, DataType.F32, new Shape(Dim.Sym("T"), 8));
        var y = b.Add(b.Unary("Abs", x), b.Unary("Abs", x));
        _ = b.Unary("Exp", x);
        b.Output("y", y);
        return new ModelDefinition
        {
            Architecture = "test_arch",
            Name = "redundant",
            Weights = new Dictionary<string, WeightInfo>(),
            Graphs = new Dictionary<string, ComputeGraph> { ["decode"] = b.Build() },
            Properties = new Dictionary<string, object>(),
        };
    }

    [Fact]
    public void OptimizesInPlaceAndKeepsArtifacts()
    {
        using var tmp = new TempDir();
        var p = tmp.File("redundant.slm");
        var set = new CompilationCache(FakeArtifactDevice.Code());
        set.Put("kernel/1", new byte[] { 1, 2, 3 });
        ModelPackage.Save(p, RedundantDefinition(), [set]);

        var result = PackageOptimizer.Optimize(p);

        Assert.True(result.Optimized);
        Assert.Equal(p, result.PackagePath);
        Assert.Equal(GraphOptimizer.PipelineId, result.PipelineId);
        Assert.True(result.NodesAfter["decode"] < result.NodesBefore["decode"],
            $"expected fewer nodes, got {result.NodesBefore["decode"]} -> {result.NodesAfter["decode"]}");
        Assert.Contains(result.Statistics, s => s.Changed);
        Assert.Equal(1, result.ArtifactSets);
        Assert.Equal(new FileInfo(p).Length, result.PackageBytes);
        Assert.False(File.Exists(p + ".optimized"), "the temporary file must be gone");

        using var file = new PackageFile(p);
        Assert.True(GraphOptimizer.IsOptimized(file.Definition));
        Assert.Equal(result.NodesAfter["decode"], file.Definition.Graphs["decode"].Nodes.Count);
        Assert.Contains("y", file.Definition.Graphs["decode"].Outputs.Keys);
        var stored = Assert.Single(file.ArtifactSets);
        Assert.True(stored.Entries.TryGet("kernel/1", out var bytes));
        Assert.Equal(new byte[] { 1, 2, 3 }, bytes.ToArray());
    }

    [Fact]
    public void AnOptimizedPackageIsLeftAloneUnlessForced()
    {
        using var tmp = new TempDir();
        var p = tmp.File("redundant.slm");
        ModelPackage.Save(p, RedundantDefinition(), []);
        Assert.True(PackageOptimizer.Optimize(p).Optimized);
        long length = new FileInfo(p).Length;

        var again = PackageOptimizer.Optimize(p);
        Assert.False(again.Optimized);
        Assert.Empty(again.Statistics);
        Assert.Equal(length, new FileInfo(p).Length);

        var forced = PackageOptimizer.Optimize(p, options: new PackageOptimizeOptions { Force = true });
        Assert.True(forced.Optimized);
        Assert.Equal(forced.NodesBefore["decode"], forced.NodesAfter["decode"]);
    }

    [Fact]
    public void WritesANewPackageAndCanDropArtifacts()
    {
        using var tmp = new TempDir();
        var p = tmp.File("redundant.slm");
        var dst = tmp.File("optimized.slm");
        var set = new CompilationCache(FakeArtifactDevice.Code());
        set.Put("kernel/1", new byte[] { 1 });
        ModelPackage.Save(p, RedundantDefinition(), [set]);
        long sourceLength = new FileInfo(p).Length;

        var result = PackageOptimizer.Optimize(p, dst, new PackageOptimizeOptions { KeepArtifacts = false });

        Assert.True(result.Optimized);
        Assert.Equal(dst, result.PackagePath);
        Assert.Equal(0, result.ArtifactSets);
        Assert.Equal(sourceLength, new FileInfo(p).Length);
        using (var source = new PackageFile(p))
        {
            Assert.False(GraphOptimizer.IsOptimized(source.Definition));
        }

        using var optimized = new PackageFile(dst);
        Assert.True(GraphOptimizer.IsOptimized(optimized.Definition));
        Assert.Empty(optimized.ArtifactSets);
    }
}
