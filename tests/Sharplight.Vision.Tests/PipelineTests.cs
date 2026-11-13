using Sharplight.Graph;
using Sharplight.Llm;
using Sharplight.Models;
using Sharplight.Package;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Vision.Tests;

/// <summary>Hermetic tests on a tiny randomly initialised Qwen3.5 + vision tower (no model files).</summary>
public class PipelineTests
{
    private static ImageInput Image(int w, int h, int seed = 5)
    {
        var rgb = new byte[w * h * 3];
        var rng = new Random(seed);
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                int o = (y * w + x) * 3;
                rgb[o] = (byte)(x * 255 / w);
                rgb[o + 1] = (byte)(y * 255 / h);
                rgb[o + 2] = (byte)(rng.Next(256));
            }
        }

        return new ImageInput(w, h, rgb);
    }

    private static float[] Encode(ModelDefinition def, DeviceSlot slot, ImageInput image)
    {
        using var enc = VisionEncoder.Create(def, [slot], limits: new ImageTokenLimits(null, 64));
        return enc.Encode(image);
    }

    [Fact]
    public void Adapter_gives_the_same_model_whether_the_tower_is_in_the_primary_or_an_auxiliary_source()
    {
        var a = Tiny.Build(visionInPrimary: true, visionAuxiliary: false);
        var b = Tiny.Build(visionInPrimary: false, visionAuxiliary: true);
        Assert.Equal(a.Graph(WellKnown.VisionEncodeGraph).Dump(), b.Graph(WellKnown.VisionEncodeGraph).Dump());
        Assert.Equal(a.Graphs.Keys.Order(), b.Graphs.Keys.Order());
        Assert.Contains(WellKnown.DecodeMultimodalGraph, a.Graphs.Keys);
        Assert.Equal(VisionParameters.Of(a), VisionParameters.Of(b), new ParamComparer());
        Assert.Equal(a.Weights.Keys.Order(), b.Weights.Keys.Order());
        // A model without a tower has no vision graphs.
        var t = Tiny.Build(visionInPrimary: false, visionAuxiliary: false);
        Assert.DoesNotContain(WellKnown.VisionEncodeGraph, t.Graphs.Keys);
        Assert.Null(VisionParameters.Of(t));
    }

    private sealed class ParamComparer : IEqualityComparer<VisionParameters?>
    {
        public bool Equals(VisionParameters? x, VisionParameters? y) => x is not null && y is not null
            && x with { ImageMean = [], ImageStd = [] } == y with { ImageMean = [], ImageStd = [] } && x.ImageMean.SequenceEqual(y.ImageMean) && x.ImageStd.SequenceEqual(y.ImageStd);
        public int GetHashCode(VisionParameters? o) => 0;
    }

    [Fact]
    public void Tower_rejects_a_language_model_of_another_width()
    {
        var primary = Tiny.Text(false);
        var aux = Tiny.VisionOnly();
        var md = new Dictionary<string, object>(aux.Metadata) { ["qwen3_5.vision.out_hidden_size"] = 48 };
        var bad = new ModelSource { Format = "test", Architecture = Tiny.Arch, Metadata = md, Weights = aux.Weights };
        Assert.Throws<InvalidDataException>(() => new Sharplight.Architectures.Qwen35Adapter().Build(
            new ModelSourceSet { Primary = primary, Auxiliary = new Dictionary<string, ModelSource> { [AuxiliaryRoles.Mmproj] = bad } }, new AdapterOptions()));
    }

    [Theory]
    [InlineData(48, 24)]    // 12×6 patches → 6×3 = 18 tokens, not square
    [InlineData(32, 32)]
    [InlineData(100, 36)]   // not a multiple of 8: resized by the preprocessor
    public void Vision_graph_gpu_matches_cpu(int w, int h)
    {
        var def = Tiny.Build(true, false);
        var cpu = Encode(def, Env.Cpu(), Image(w, h));
        var gpu = Env.Vulkan();
        Assert.SkipUnless(gpu is not null, "no Vulkan device");
        var vk = Encode(def, gpu!, Image(w, h));
        var (abs, rel, rms) = Env.Compare(vk, cpu);
        Console.Error.WriteLine($"tiny tower {w}x{h}: Vulkan vs CPU max abs {abs:E2}, rel RMS {rms:E2}");
        Assert.True(rms < 1e-3, $"relative RMS {rms}");
    }

    [Fact]
    public void Vision_output_has_one_row_per_merged_patch_and_is_deterministic()
    {
        var def = Tiny.Build(true, false);
        var a = Encode(def, Env.Cpu(), Image(48, 24));
        Assert.Equal(18 * Tiny.Hidden, a.Length);
        Assert.Equal(a, Encode(def, Env.Cpu(), Image(48, 24)));
        Assert.NotEqual(a, Encode(def, Env.Cpu(), Image(48, 24, seed: 6)));
        Assert.All(a, v => Assert.True(float.IsFinite(v)));
    }

    /// <summary>decode_multimodal with a zero mask equals decode; with the token embedding rows as embeddings and a full mask too.</summary>
    [Fact]
    public void Multimodal_decoder_equals_token_decoder_for_equivalent_inputs()
    {
        var def = Tiny.Build(true, false);
        var tokens = new[] { 3, 17, 5, 9, 21 };
        var table = def.Weights["token_embedding.weight"].ReadAll();
        var emb = new float[tokens.Length * Tiny.Hidden];
        for (int t = 0; t < tokens.Length; t++)
        {
            Buffer.BlockCopy(table, tokens[t] * Tiny.Hidden * 4, emb, t * Tiny.Hidden * 4, Tiny.Hidden * 4);
        }

        float[] Run(string graphName, Action<CompiledModel, ComputeGraph> setup)
        {
            var g = def.Graph(graphName);
            using var cm = CompiledModel.Compile(g, new PlacementOptions { Devices = [Env.Cpu()], Compile = new CompileOptions { MaxTokens = 16, ContextLength = 32 } });
            int T = tokens.Length;
            cm.SetInput<int>(g.Input(InputRole.Positions), Enumerable.Range(0, T).SelectMany(t => new[] { t, t, t, 0 }).ToArray());
            cm.SetInput<int>(g.Input(InputRole.OutputRows), [T - 1]);
            setup(cm, g);
            var run = new RunParameters();
            run.Symbols["T"] = T;
            run.Symbols["R"] = 1;
            run.Scalars["n_past"] = 0;
            cm.Run(run);
            var logits = new float[Tiny.Vocab];
            cm.GetOutput<float>("logits", logits);
            return logits;
        }

        var reference = Run(WellKnown.DecodeGraph, (cm, g) => cm.SetInput<int>(g.Input(InputRole.TokenIds), tokens));
        var textOnly = Run(WellKnown.DecodeMultimodalGraph, (cm, g) =>
        {
            cm.SetInput<int>(g.Input(InputRole.TokenIds), tokens);
            cm.SetInput<float>(g.Input(InputRole.Embeddings), new float[emb.Length]);
            cm.SetInput<float>(g.Input(InputRole.EmbeddingMask), new float[tokens.Length]);
        });
        var embedded = Run(WellKnown.DecodeMultimodalGraph, (cm, g) =>
        {
            cm.SetInput<int>(g.Input(InputRole.TokenIds), new int[tokens.Length]);
            cm.SetInput<float>(g.Input(InputRole.Embeddings), emb);
            cm.SetInput<float>(g.Input(InputRole.EmbeddingMask), Enumerable.Repeat(1f, tokens.Length).ToArray());
        });
        Assert.Equal(reference, textOnly);
        var (abs, _, rms) = Env.Compare(embedded, reference);
        Assert.True(rms < 1e-6, $"embedding rows vs token rows: rel RMS {rms} (max abs {abs})");
    }

    [Fact]
    public void Package_round_trip_keeps_the_vision_tower()
    {
        var def = Tiny.Build(true, false);
        var tmp = Path.Combine(Path.GetTempPath(), "slvision-" + Guid.NewGuid().ToString("N")[..8] + ModelPackage.Extension);
        try
        {
            ModelPackage.Save(tmp, def, []);
            var src = new ModelPackageImporter().Open(tmp);
            using var _ = new Disposer(src.OwnedResources);
            var def2 = ModelAdapterRegistry.Build(new ModelSourceSet { Primary = src });
            foreach (var name in new[] { WellKnown.VisionEncodeGraph, WellKnown.DecodeMultimodalGraph })
            {
                var g = def.Graph(name);
                var g2 = def2.Graph(name);
                string Sig(ComputeGraph x) => string.Join("\n", x.Nodes.Select(n => $"{n.Op.Name} {n.Attributes} @{n.Group} -> {string.Join(";", n.Outputs.Select(o => $"{o.Type}{o.Shape}"))}"));
                Assert.Equal(Sig(g), Sig(g2));
                Assert.Equal(g.Inputs.Select(i => $"{i.Input}{i.Shape}"), g2.Inputs.Select(i => $"{i.Input}{i.Shape}"));
            }
            Assert.Equal(VisionParameters.Of(def), VisionParameters.Of(def2), new ParamComparer());
            // The architecture's prompt assembly is found by id, no adapter involved.
            BuiltinAdapters.RegisterAll();
            Assert.NotNull(MultimodalRegistry.Find(def2.Architecture));
            var img = Image(48, 24);
            Assert.Equal(Encode(def, Env.Cpu(), img), Encode(def2, Env.Cpu(), img));
        }
        finally { File.Delete(tmp); }
    }

    [Fact]
    public void Reshape_with_outer_symbol_merges_rows()
    {
        var b = new GraphBuilder("t", _ => null);
        var x = b.Input("x", InputRole.Other, DataType.F32, new Shape(Dim.Sym("N"), 3));
        var y = b.Emit1("Reshape", [x], Attributes.From(("shape", new long[] { 0, 6 }), ("outer_symbol", "M")));
        Assert.Equal("M", y.Shape.Symbol);
        b.Output("y", b.Add(y, y));
        var g = b.Build();
        foreach (var slot in new[] { Env.Cpu(), Env.Vulkan() })
        {
            if (slot is null)
            {
                continue;
            }

            using var cm = CompiledModel.Compile(g, new PlacementOptions { Devices = [slot], Compile = new CompileOptions { MaxTokens = 16, SymbolBounds = new Dictionary<string, long> { ["N"] = 16, ["M"] = 8 } } });
            cm.SetInput<float>(g.Inputs[0], Enumerable.Range(0, 24).Select(i => (float)i).ToArray());
            var run = new RunParameters();
            run.Symbols["N"] = 8;
            run.Symbols["M"] = 4;
            cm.Run(run);
            var o = new float[24];
            cm.GetOutput<float>("y", o);
            Assert.Equal(Enumerable.Range(0, 24).Select(i => 2f * i), o);
        }
        Assert.Throws<ArgumentException>(() => new GraphBuilder("t", _ => null).Emit1("Reshape",
            [new GraphBuilder("u", _ => null).Input("x", InputRole.Other, DataType.F32, new Shape(4, 3))], Attributes.From(("shape", new long[] { 0, 6 }), ("outer_symbol", "M"))));
    }
}
