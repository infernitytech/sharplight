using Sharplight.Cpu;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Onnx;
using Sharplight.Runtime;
using Sharplight.Tensors;
using Xunit;

namespace Sharplight.Package.Tests;

public sealed class TempDir : IDisposable
{
    public string Path { get; } = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "sharplight-pkg-" + Guid.NewGuid().ToString("N")[..8]);
    public TempDir() => Directory.CreateDirectory(Path);
    public string File(string name) => System.IO.Path.Combine(Path, name);
    public void Dispose() { try { Directory.Delete(Path, true); } catch { } }
}

public class PackageTests
{
    private static readonly Lazy<IComputeDevice> s_cpu = new(() => { var p = new CpuExecutionProvider(); return p.OpenDevice(p.GetDevices()[0]); });
    private static string OnnxDir => Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models", "onnx");

    /// <summary>A model built from ONNX survives save/load: same graph structure, bit-identical outputs.</summary>
    [Theory]
    [InlineData("mnist-12")] [InlineData("squeezenet1.1-7")] [InlineData("mobilenetv2-12")] [InlineData("deit-tiny")] [InlineData("minilm-l6")]
    public void OnnxModelRoundTrip(string name)
    {
        var path = Path.Combine(OnnxDir, name + ".onnx");
        if (!File.Exists(path))
        {
            Assert.Skip($"{path} not present");
        }

        using var tmp = new TempDir();
        var opts = new OnnxImportOptions();
        if (name == "deit-tiny")
        {
            opts.InputShapes["pixel_values"] = [-1, 3, 224, 224];
        }

        if (name == "minilm-l6")
        {
            foreach (var i in new[] { "input_ids", "attention_mask", "token_type_ids" })
            {
                opts.InputShapes[i] = [-1, 16];
            }
        }

        var def = OnnxGraphAdapter.Build(OnnxImporter.Open(path, opts));
        try
        {
            var pkgPath = tmp.File(name + ModelPackage.Extension);
            ModelPackage.Save(pkgPath, def, []);
            Assert.True(ModelPackage.IsPackage(pkgPath));
            var src = new ModelPackageImporter().Open(pkgPath);
            var def2 = ModelAdapterRegistry.Build(new ModelSourceSet { Primary = src });
            try
            {
                Assert.Equal(def.Architecture, def2.Architecture);
                var g = def.Graph(OnnxGraphAdapter.MainGraph);
                var g2 = def2.Graph(OnnxGraphAdapter.MainGraph);
                Assert.Equal(g.Nodes.Select(n => $"{n.Op.Name} {n.Attributes} @{n.Group}"), g2.Nodes.Select(n => $"{n.Op.Name} {n.Attributes} @{n.Group}"));
                Assert.Equal(g.Nodes.Select(n => string.Join(";", n.Outputs.Select(o => $"{o.Type}{o.Shape}"))), g2.Nodes.Select(n => string.Join(";", n.Outputs.Select(o => $"{o.Type}{o.Shape}"))));
                Assert.Equal(g.Outputs.Keys.Order(), g2.Outputs.Keys.Order());
                Assert.Equal(def.Properties.Keys.Order(), def2.Properties.Keys.Order());

                var a = Run(g, def);
                var b = Run(g2, def2);
                Assert.Equal(a.Count, b.Count);
                for (int i = 0; i < a.Count; i++)
                {
                    Assert.True(a[i].AsSpan().SequenceEqual(b[i]), $"output {i} differs after round trip");
                }
            }
            finally { foreach (var r in def2.OwnedResources)
                {
                    r.Dispose();
                }
            }
        }
        finally { foreach (var r in def.OwnedResources)
            {
                r.Dispose();
            }
        }
    }

    private static List<byte[]> Run(ComputeGraph g, ModelDefinition def)
    {
        var bounds = new Dictionary<string, long> { [OnnxGraphAdapter.BatchSymbol] = 1 };
        using var cm = CompiledModel.Compile(g, new PlacementOptions
        {
            Devices = [new DeviceSlot { Device = s_cpu.Value }],
            Compile = new CompileOptions { MaxTokens = 16, SymbolBounds = bounds },
        });
        var run = new RunParameters();
        foreach (var (k, v) in bounds)
        {
            run.Symbols[k] = v;
        }

        run.Symbols["T"] = 16;
        var rng = new Random(7);
        foreach (var input in g.Inputs)
        {
            long n = input.Shape.Elements(run.Symbols);
            if (input.Type == DataType.I32)
            {
                cm.SetInput<int>(input, Enumerable.Range(0, (int)n).Select(_ => rng.Next(0, 2)).ToArray()); // valid as ids, masks and type ids
            }
            else if (input.Type == DataType.I64)
            {
                cm.SetInput<long>(input, Enumerable.Range(0, (int)n).Select(_ => (long)rng.Next(0, 2)).ToArray()); // valid as ids, masks and type ids
            }
            else
            {
                cm.SetInput<float>(input, Enumerable.Range(0, (int)n).Select(_ => (float)rng.NextDouble()).ToArray());
            }
        }
        cm.Run(run);
        var outs = new List<byte[]>();
        foreach (var (name, v) in g.Outputs)
        {
            var buf = new byte[v.Type.ByteSize(v.Shape.Elements(run.Symbols))];
            cm.GetOutput(name, buf);
            outs.Add(buf);
        }
        return outs;
    }

    private static ModelDefinition SmallDefinition(Dictionary<string, object>? extraMeta = null)
    {
        var w = WeightInfo.Derived("w", DataType.F32, new Shape(4, 8), () => new byte[4 * 8 * 4].Select((_, i) => (byte)i).ToArray());
        var weights = new Dictionary<string, WeightInfo> { ["w"] = w };
        var state = new StateDefinition { Name = "kc", Type = DataType.F16, Shape = new Shape(Dim.Sym("ctx"), 4), Kind = StateKind.KeyCache, Group = "layers.0" };
        var b = new GraphBuilder("decode", n => weights.GetValueOrDefault(n));
        var x = b.Input("x", InputRole.Embeddings, DataType.F32, new Shape(Dim.Sym("T"), 8));
        Value y;
        using (b.Group("layers.0"))
        {
            y = b.MatMul(x, b.Weight("w"), "proj");
            y = b.Add(y, b.Constant("bias", DataType.F32, new Shape(4), new byte[16]));
            b.CacheWrite(b.State(state), y);
        }
        b.Output("y", y);
        var meta = new Dictionary<string, object>
        {
            ["tokenizer.tokens"] = new[] { "a", "b", "ü" },
            ["tokenizer.token_type"] = new[] { 1, 2, 3 },
            ["x.f"] = 1.5f, ["x.d"] = 2.25, ["x.l"] = -3L, ["x.u"] = 4u, ["x.b"] = true, ["x.dt"] = DataType.BF16,
            ["x.objs"] = new object[] { 1L, 2L },
        };
        foreach (var (k, v) in extraMeta ?? [])
        {
            meta[k] = v;
        }

        return new ModelDefinition
        {
            Architecture = "test_arch",
            Name = "small",
            Weights = weights,
            Graphs = new Dictionary<string, ComputeGraph> { ["decode"] = b.Build() },
            States = [state],
            Properties = new Dictionary<string, object> { [WellKnown.EosTokenIds] = new[] { 1, 2 }, [WellKnown.VocabSize] = 3 },
            SourceMetadata = meta,
        };
    }

    [Fact]
    public void DefinitionRoundTripKeepsMetadataStatesAndWeights()
    {
        using var tmp = new TempDir();
        var def = SmallDefinition();
        var p = tmp.File("small.slm");
        ModelPackage.Save(p, def, []);
        using var f = new PackageFile(p);
        var d = f.Definition;
        Assert.Equal("test_arch", d.Architecture);
        Assert.Equal("small", d.Name);
        Assert.Equal(new[] { "a", "b", "ü" }, (string[])d.SourceMetadata["tokenizer.tokens"]);
        Assert.Equal(new[] { 1, 2, 3 }, (int[])d.SourceMetadata["tokenizer.token_type"]);
        Assert.Equal(1.5f, d.SourceMetadata["x.f"]);
        Assert.Equal(2.25, d.SourceMetadata["x.d"]);
        Assert.Equal(-3L, d.SourceMetadata["x.l"]);
        Assert.Equal(4u, d.SourceMetadata["x.u"]);
        Assert.Equal(true, d.SourceMetadata["x.b"]);
        Assert.Same(DataType.BF16, d.SourceMetadata["x.dt"]);
        Assert.Equal(new[] { 1L, 2L }, (long[])d.SourceMetadata["x.objs"]);
        Assert.Equal(new[] { 1, 2 }, (int[])d.Properties[WellKnown.EosTokenIds]);

        var s = Assert.Single(d.States);
        Assert.Equal(("kc", StateKind.KeyCache, "layers.0", "f16[ctx,4]"), (s.Name, s.Kind, s.Group, $"{s.Type}{s.Shape}".Replace(" ", "")));

        var w = d.Weights["w"];
        Assert.Equal(def.Weights["w"].ReadAll(), w.ReadAll());
        Assert.Equal(0, w.Source.TryGetPointer() % 4096); // 64 KiB file offset; mmap bases are page aligned

        var g = d.Graph("decode");
        Assert.Equal(["MatMul", "Add", "CacheWrite"], g.Nodes.Select(n => n.Op.Name));
        Assert.Equal("kc", Assert.Single(g.Nodes[2].Writes).Name);
        Assert.Contains("n_past", g.RuntimeScalars);
        Assert.Equal("proj", g.Nodes[0].Output.Name);
    }

    [Fact]
    public void UnsupportedMetadataValuesAreRejected()
    {
        using var tmp = new TempDir();
        var def = SmallDefinition(new() { ["bad"] = new Uri("http://x") });
        var e = Assert.Throws<NotSupportedException>(() => ModelPackage.Save(tmp.File("bad.slm"), def, []));
        Assert.Contains("bad", e.Message);
        Assert.False(File.Exists(tmp.File("bad.slm")));
        Assert.False(File.Exists(tmp.File("bad.slm.tmp")));
    }

    [Fact]
    public void TargetsAreStoredReplacedAndAppended()
    {
        using var tmp = new TempDir();
        var p = tmp.File("t.slm");
        var a = new CompilationCache("dev-a");
        a.Put("kernel/1", new byte[] { 1, 2, 3 });
        a.Put("vulkan/pipeline_cache", new byte[100]);
        ModelPackage.Save(p, SmallDefinition(), [a, new CompilationCache("empty")]);

        using (var f = new PackageFile(p))
        {
            Assert.Equal(["dev-a"], f.Targets);
            Assert.True(f.Find("dev-a")!.TryGet("kernel/1", out var k));
            Assert.Equal(new byte[] { 1, 2, 3 }, k.ToArray());
            Assert.False(f.Find("dev-a")!.TryGet("kernel/2", out _));
            Assert.Null(f.Find("dev-b"));

            // Append while the file is open: the open mapping keeps serving its data.
            var a2 = new CompilationCache("dev-a");
            a2.Put("kernel/1", new byte[] { 9 });
            var b = new CompilationCache("dev-b");
            b.Put("kernel/7", new byte[] { 7, 7 });
            ModelPackage.AddTargets(p, [a2, b]);
            Assert.True(f.Find("dev-a")!.TryGet("kernel/1", out var still));
            Assert.Equal(new byte[] { 1, 2, 3 }, still.ToArray());
        }

        using (var f = new PackageFile(p))
        {
            Assert.Equal(["dev-a", "dev-b"], f.Targets.Order());
            Assert.True(f.Find("dev-a")!.TryGet("kernel/1", out var k));
            Assert.Equal(new byte[] { 9 }, k.ToArray());
            Assert.True(f.Find("dev-b")!.TryGet("kernel/7", out var k7));
            Assert.Equal(new byte[] { 7, 7 }, k7.ToArray());
            Assert.Equal(SmallDefinition().Weights["w"].ReadAll(), f.Weights["w"].ReadAll());
        }
    }

    [Fact]
    public void RecordingCacheCapturesFallbackHitsAndNewEntries()
    {
        var stored = new CompilationCache("t");
        stored.Put("old", new byte[] { 1 });
        var c = new CompilationCache("t", stored);
        Assert.True(c.TryGet("old", out _));
        c.Put("new", new byte[] { 2 });
        Assert.False(c.TryGet("missing", out _));
        Assert.Equal(["new", "old"], c.Entries.Keys.Order());
    }

    [Fact]
    public void NotAPackage()
    {
        using var tmp = new TempDir();
        var p = tmp.File("x.slm");
        File.WriteAllBytes(p, new byte[100]);
        Assert.False(ModelPackage.IsPackage(p));
        Assert.Throws<InvalidDataException>(() => new PackageFile(p));
    }
}
