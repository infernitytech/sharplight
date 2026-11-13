using Sharplight.Graph;
using Sharplight.Graph.Optimization;
using Sharplight.Models;
using Sharplight.Tensors;
using static Sharplight.Cpu.Tests.Optimization.OptimizationKit;

namespace Sharplight.Cpu.Tests.Optimization;

public class GraphOptimizerTests
{
    private static ComputeGraph Redundant(string name, out Value x)
    {
        var tg = new TestGraph(name);
        x = tg.InF("x", TestGraph.T, 4);
        var c = ConstF(tg.B, "c", [1, 2, 3, 4], 4);
        var folded = tg.B.Mul(c, c);
        var y = tg.B.Add(tg.B.Unary("Abs", x), tg.B.Unary("Abs", x));
        _ = tg.B.Unary("Exp", x);
        tg.B.Output("y", tg.B.Add(tg.B.Reshape(tg.B.Reshape(y, 0, 2, 2), 0, 4), folded));
        return tg.B.Build();
    }

    private static ModelDefinition Definition(params ComputeGraph[] graphs) => new()
    {
        Architecture = "test",
        Weights = new Dictionary<string, WeightInfo>(),
        Graphs = graphs.ToDictionary(g => g.Name),
        Properties = new Dictionary<string, object> { ["answer"] = 42L },
    };

    [Fact]
    public void OptimizesEveryGraphRecordsThePipelineAndReportsStatistics()
    {
        var a = Redundant("a", out var xa);
        var b = Redundant("b", out _);
        var definition = Definition(a, b);

        var result = GraphOptimizer.Default.Optimize(definition);

        Assert.Equal(GraphOptimizer.PipelineId, result.Definition.Properties[GraphOptimizer.OptimizedProperty]);
        Assert.Equal(42L, result.Definition.Properties["answer"]);
        Assert.False(definition.Properties.ContainsKey(GraphOptimizer.OptimizedProperty));
        Assert.True(GraphOptimizer.IsOptimized(result.Definition));
        Assert.Equal(a.Nodes.Count, result.NodesBefore["a"]);
        Assert.Equal(3, result.NodesAfter["a"]);    // Abs, Add, Add: duplicates merged, dead Exp dropped, reshapes collapsed to an identity, Mul folded
        Assert.Equal(3, result.NodesAfter["b"]);
        var optimizedA = result.Definition.Graphs["a"];
        Assert.Equal(Enumerable.Range(0, optimizedA.Nodes.Count), optimizedA.Nodes.Select(n => n.Id));   // ids stay positions
        Assert.Contains(result.Statistics, s => s.Graph == "a" && s.Pass == "dead-nodes" && s.Changed);
        Assert.All(result.Statistics, s => Assert.InRange(s.Round, 1, GraphOptimizer.MaxRounds));
        AssertSameResults(a, result.Definition.Graphs["a"], new Dictionary<Value, byte[]> { [xa] = Bytes(Kit.Rand(12, 1)) }, Kit.P(3));
    }

    [Fact]
    public void PrepareForLoadRunsOnceAndCanBeDisabled()
    {
        var definition = Definition(Redundant("a", out _));

        var optimized = GraphOptimizer.PrepareForLoad(definition);
        Assert.NotSame(definition, optimized);
        Assert.Same(optimized, GraphOptimizer.PrepareForLoad(optimized));
        Assert.Same(definition, GraphOptimizer.PrepareForLoad(definition, enabled: false));

        var older = Definition(Redundant("a", out _));
        var tagged = new ModelDefinition
        {
            Architecture = older.Architecture, Weights = older.Weights, Graphs = older.Graphs,
            Properties = new Dictionary<string, object> { [GraphOptimizer.OptimizedProperty] = "sharplight-ir/0" },
        };
        Assert.NotSame(tagged, GraphOptimizer.PrepareForLoad(tagged));

        var previous = Environment.GetEnvironmentVariable(GraphOptimizer.DisableEnvironmentVariable);
        try
        {
            Environment.SetEnvironmentVariable(GraphOptimizer.DisableEnvironmentVariable, "1");
            Assert.Same(definition, GraphOptimizer.PrepareForLoad(definition));
        }
        finally
        {
            Environment.SetEnvironmentVariable(GraphOptimizer.DisableEnvironmentVariable, previous);
        }
    }

    [Fact]
    public void OptimizesTheBranchesOfControlFlow()
    {
        static ComputeGraph Branch(string name, bool redundant)
        {
            var b = new GraphBuilder(name, _ => null);
            var v = b.Input("v", InputRole.Other, DataType.F32, new Shape(4));
            var y = redundant ? b.Add(b.Unary("Abs", v), b.Unary("Abs", v)) : b.Unary("Neg", v);
            if (redundant)
            {
                _ = b.Unary("Exp", v);
            }

            b.Output("out0", y);
            return b.Build();
        }

        var tg = new TestGraph();
        var condition = tg.InF("cond", 1);
        var x = tg.InF("x", 4);
        var then = Branch("then", redundant: true);
        var attributes = Attributes.From(("inputs", new[] { "v" }), ("outputs", new[] { "out0" }), ("output_symbols", new[] { "" }),
            ("then_branch", then), ("else_branch", Branch("else", redundant: false)));
        var result = tg.B.Emit("If", [condition, x], attributes)[0];
        tg.B.Output("y", tg.B.Unary("Neg", result));
        var g = tg.B.Build();

        var optimized = GraphOptimizer.Default.Optimize(g);

        var optimizedThen = optimized.Graph.Nodes.Single(n => n.Op.Name == "If").Attributes.Get<ComputeGraph>("then_branch");
        Assert.Equal(["Abs", "Add"], optimizedThen.Nodes.Select(n => n.Op.Name));
        Assert.Equal(4, then.Nodes.Count);
        Assert.Contains(optimized.Statistics, s => s.Graph.StartsWith("t/node#", StringComparison.Ordinal) && s.Changed);
        foreach (float c in new[] { 1f, 0f })
        {
            AssertSameResults(g, optimized.Graph, new Dictionary<Value, byte[]> { [condition] = Bytes([c]), [x] = Bytes(Kit.Rand(4, 2)) });
        }
    }
}
