using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;
using Xunit;

namespace Sharplight.Package.Tests;

public class IrListingTests
{
    /// <summary>The IR of a stored graph lists declarations and nodes in the dump format, with value references per line.</summary>
    [Fact]
    public void StoredGraphIrMatchesTheDumpAndReferencesValues()
    {
        using var tmp = new TempDir();
        var p = tmp.File("small.slm");
        var def = PackageTests.SmallDefinition();
        ModelPackage.Save(p, def, []);
        var graph = ModelPackageReader.Open(p).Definition.Graphs.Single();

        var dump = def.Graph("decode").Dump().Split('\n', StringSplitOptions.RemoveEmptyEntries);
        Assert.Equal(dump, graph.Ir.Select(l => new string(' ', 2 * l.Depth) + l.Text));
        Assert.Contains(graph.Ir, l => l.Kind == GraphDumpLineKind.Weight && l.Text.StartsWith("weight %", StringComparison.Ordinal));
        Assert.Contains(graph.Ir, l => l.Kind == GraphDumpLineKind.State && l.Text.EndsWith("(KeyCache)", StringComparison.Ordinal));
        Assert.Contains(graph.Ir, l => l.Kind == GraphDumpLineKind.Constant);

        var matMul = graph.Ir.Single(l => l.Text.Contains("MatMul(", StringComparison.Ordinal));
        var input = graph.Ir.Single(l => l.Kind == GraphDumpLineKind.Input);
        Assert.Contains(input.Defines[0], matMul.Uses);
        Assert.Equal(graph.Nodes.Single(n => n.Op == "MatMul").Id, matMul.NodeId);
        Assert.All(graph.Ir, l => Assert.Equal(0, l.Scope));
    }

    /// <summary>Control-flow subgraphs are listed below their node, indented, each in its own value scope.</summary>
    [Fact]
    public void IfBranchesAreListedAsNestedScopes()
    {
        var b = new GraphBuilder("main", _ => null);
        var cond = b.Input("cond", InputRole.Other, DataType.I32, new Shape(1));
        var x = b.Input("x", InputRole.Other, DataType.F32, new Shape(4));
        var attributes = Attributes.From(
            ("inputs", new[] { "a" }),
            ("outputs", new[] { "out0" }),
            ("output_symbols", new[] { "" }),
            ("then_branch", Branch("then", "Silu")),
            ("else_branch", Branch("else", "Sigmoid")));
        var result = b.Emit("If", [cond, x], attributes, "choice");
        b.Output("y", result[0]);

        var lines = IrLine.Build(b.Build());
        var ifLine = lines.Single(l => l.Kind == GraphDumpLineKind.Node && l.Text.Contains("If(", StringComparison.Ordinal));
        Assert.Contains("then_branch=<graph then>", ifLine.Text, StringComparison.Ordinal);
        var headers = lines.Where(l => l.Kind == GraphDumpLineKind.Graph).ToList();
        Assert.Equal(["graph main", "else_branch: graph else", "then_branch: graph then"], headers.Select(h => h.Text));
        Assert.Equal([0, 1, 2], headers.Select(h => h.Scope));
        Assert.All(headers.Skip(1), h => Assert.Equal(ifLine.Depth + 1, h.Depth));
        var silu = lines.Single(l => l.Text.Contains("Silu(", StringComparison.Ordinal));
        Assert.Equal(ifLine.Depth + 2, silu.Depth);
        var branchInput = lines.Single(l => l.Kind == GraphDumpLineKind.Input && l.Scope == silu.Scope);
        Assert.Equal(branchInput.Defines, silu.Uses);
    }

    private static ComputeGraph Branch(string name, string op)
    {
        var b = new GraphBuilder(name, _ => null);
        var a = b.Input("a", InputRole.Other, DataType.F32, new Shape(4));
        b.Output("out0", b.Unary(op, a));
        return b.Build();
    }
}
