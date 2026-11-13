using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Package;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.PackageViewer.Tests;

/// <summary>Writes small packages for the viewer tests with <see cref="ModelPackage.Save"/>.</summary>
public static class TestPackages
{
    public const string CodeKernelKey = "kernel/0001";

    /// <summary>
    /// A package with one graph (<c>decode</c>, 4 nodes over 2 layers), two weights, a state, properties, metadata and two Vulkan
    /// artifact sets (code tier with two kernels, binary tier with a pipeline cache).
    /// </summary>
    public static void WriteSmall(string path)
    {
        var weights = new Dictionary<string, WeightInfo>
        {
            ["w0"] = Weight("w0", DataType.F32, new Shape(4, 8)),
            ["w1"] = Weight("w1", DataType.F16, new Shape(4, 4)),
        };
        var state = new StateDefinition { Name = "kc", Type = DataType.F16, Shape = new Shape(Dim.Sym("ctx"), 4), Kind = StateKind.KeyCache, Group = "layers.0" };
        var b = new GraphBuilder("decode", n => weights.GetValueOrDefault(n));
        var x = b.Input("x", InputRole.Embeddings, DataType.F32, new Shape(Dim.Sym("T"), 8));
        Value y;
        using (b.Group("layers.0"))
        {
            y = b.MatMul(x, b.Weight("w0"), "proj");
            b.CacheWrite(b.State(state), y);
        }

        using (b.Group("layers.1"))
        {
            y = b.Silu(b.MatMul(y, b.Weight("w1"), "proj2"), "act");
        }

        b.Output("y", y);
        var definition = new ModelDefinition
        {
            Architecture = "test_arch",
            Name = "small",
            Weights = weights,
            Graphs = new Dictionary<string, ComputeGraph> { ["decode"] = b.Build() },
            States = [state],
            Properties = new Dictionary<string, object> { [WellKnown.VocabSize] = 3 },
            SourceMetadata = new Dictionary<string, object>
            {
                ["tokenizer.tokens"] = new[] { "a", "b", "c" },
                ["general.name"] = "small test model",
            },
        };

        var code = new CompilationCache(Descriptor(ArtifactTiers.Code, "spirv", (ArtifactDescriptorKeys.Target, "spirv_1_6")));
        code.Put(CodeKernelKey, new byte[64]);
        code.Put("kernel/0002", new byte[32]);
        var binary = new CompilationCache(Descriptor(ArtifactTiers.Binary, "pipeline-cache", (ArtifactDescriptorKeys.Vendor, "10de")));
        binary.Put("vulkan/pipeline_cache", new byte[100]);
        ModelPackage.Save(path, definition, [code, binary]);
    }

    /// <summary>A package with one graph <c>main</c> of <paramref name="nodeCount"/> elementwise nodes (100 layers).</summary>
    public static void WriteLarge(string path, int nodeCount)
    {
        var b = new GraphBuilder("main", _ => null);
        var y = b.Input("x", InputRole.Embeddings, DataType.F32, new Shape(Dim.Sym("T"), 64));
        int perLayer = Math.Max(1, nodeCount / 100);
        for (int i = 0; i < nodeCount; i++)
        {
            using (b.Group($"layers.{i / perLayer}"))
            {
                y = i % 2 == 0 ? b.Silu(y) : b.Add(y, y);
            }
        }

        b.Output("y", y);
        var definition = new ModelDefinition
        {
            Architecture = "test_large",
            Name = "large",
            Weights = new Dictionary<string, WeightInfo>(),
            Graphs = new Dictionary<string, ComputeGraph> { ["main"] = b.Build() },
            States = [],
            Properties = new Dictionary<string, object>(),
            SourceMetadata = new Dictionary<string, object>(),
        };
        ModelPackage.Save(path, definition, []);
    }

    private static WeightInfo Weight(string name, DataType type, Shape shape) =>
        WeightInfo.Derived(name, type, shape, () => new byte[type.ByteSize(shape.Elements())]);

    private static Dictionary<string, string> Descriptor(string tier, string kind, params (string Key, string Value)[] more)
    {
        var descriptor = new Dictionary<string, string>
        {
            [ArtifactDescriptorKeys.Provider] = "vulkan",
            [ArtifactDescriptorKeys.Tier] = tier,
            [ArtifactDescriptorKeys.Kind] = kind,
        };
        foreach (var (key, value) in more)
        {
            descriptor[key] = value;
        }

        return descriptor;
    }
}
