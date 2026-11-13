using System.Runtime.InteropServices;
using Sharplight.Architectures;
using Sharplight.Models;
using Sharplight.Tensors;
using Xunit;

namespace Sharplight.HuggingFace.Tests;

/// <summary>Graph-level checks of the generic decoder adapter on a tiny synthetic model: longrope factor selection and sliding windows.</summary>
public class DecoderGraphTests
{
    private sealed class Mem(byte[] data) : IWeightSource
    {
        public long ByteLength => data.Length;
        public void Read(long offset, Span<byte> destination) => data.AsSpan((int)offset, destination.Length).CopyTo(destination);
    }

    private static WeightInfo W(string name, params long[] dims)
    {
        long n = dims.Aggregate(1L, (a, b) => a * b);
        var f = new float[n];
        for (int i = 0; i < n; i++)
        {
            f[i] = 0.01f * (i % 7 + 1);
        }

        return new WeightInfo { Name = name, Type = DataType.F32, Shape = new Shape(dims.Select(d => (Dim)d)), Source = new Mem(MemoryMarshal.AsBytes(f.AsSpan()).ToArray()) };
    }

    private const int Hidden = 8, Heads = 2, Kv = 1, D = 4, Ffn = 8, Vocab = 10, Layers = 2;

    private static ModelSource Source(Dictionary<string, object>? extra = null, bool longrope = false)
    {
        var md = new Dictionary<string, object>
        {
            ["decoder.layer_count"] = Layers, ["decoder.hidden_size"] = Hidden, ["decoder.ffn_hidden_size"] = Ffn,
            ["decoder.attention.head_count"] = Heads, ["decoder.attention.kv_head_count"] = Kv, ["decoder.attention.head_dim"] = D,
            ["decoder.rope.dim"] = D, ["decoder.context_length"] = 32768,
        };
        if (extra is not null)
        {
            foreach (var kv in extra)
            {
                md[kv.Key] = kv.Value;
            }
        }

        var w = new Dictionary<string, WeightInfo>();
        void Add(WeightInfo x) => w[x.Name] = x;
        Add(W("token_embedding.weight", Vocab, Hidden)); Add(W("output_norm.weight", Hidden));
        for (int i = 0; i < Layers; i++)
        {
            string p = $"layers.{i}.";
            Add(W(p + "input_norm.weight", Hidden)); Add(W(p + "post_attention_norm.weight", Hidden));
            Add(W(p + "attention.q.weight", Heads * D, Hidden)); Add(W(p + "attention.k.weight", Kv * D, Hidden));
            Add(W(p + "attention.v.weight", Kv * D, Hidden)); Add(W(p + "attention.o.weight", Hidden, Heads * D));
            Add(W(p + "mlp.gate.weight", Ffn, Hidden)); Add(W(p + "mlp.up.weight", Ffn, Hidden)); Add(W(p + "mlp.down.weight", Hidden, Ffn));
        }
        if (longrope) { Add(W("rope.freq_factors", D / 2)); Add(W("rope.freq_factors_long", D / 2)); }
        return new ModelSource { Format = "test", Architecture = "decoder", Metadata = md, Weights = w };
    }

    private static ModelDefinition Build(ModelSource src, int context = 0)
        => new DecoderAdapter().Build(new ModelSourceSet { Primary = src }, new AdapterOptions { ContextLength = context });

    private static IEnumerable<Sharplight.Graph.Node> Nodes(ModelDefinition d, string op) => d.Graph(WellKnown.DecodeGraph).Nodes.Where(n => n.Op.Name == op);

    [Theory]
    [InlineData(4096, "rope.freq_factors")]        // at the original context: short factors
    [InlineData(2048, "rope.freq_factors")]
    [InlineData(0, "rope.freq_factors_long")]      // unknown: the model's maximum (32768) > original
    [InlineData(8192, "rope.freq_factors_long")]   // configured context beyond the original: long factors
    public void LongRopeSelectsFactorsByContext(int context, string expected)
    {
        var def = Build(Source(new() { ["decoder.rope.original_context_length"] = 4096 }, longrope: true), context);
        var ropes = Nodes(def, "Rope").ToList();
        Assert.Equal(Layers * 2, ropes.Count);
        foreach (var r in ropes)
        {
            Assert.Equal(expected, r.Inputs[2].Name);
        }
    }

    [Fact]
    public void NoOriginalContextMeansShortFactorsOnly()
    {
        var def = Build(Source(longrope: true), 100000);
        foreach (var r in Nodes(def, "Rope"))
        {
            Assert.Equal("rope.freq_factors", r.Inputs[2].Name);
        }
    }

    [Theory]
    [InlineData(null, 0)]
    [InlineData(0, 0)]
    [InlineData(4096, 4096)]
    public void SlidingWindowAppliesToEveryLayerWhenDeclared(int? window, int expected)
    {
        var extra = new Dictionary<string, object>();
        if (window is { } w)
        {
            extra["decoder.attention.sliding_window"] = w;
        }

        var def = Build(Source(extra));
        var attn = Nodes(def, "CachedAttention").ToList();
        Assert.Equal(Layers, attn.Count);
        foreach (var a in attn)
        {
            Assert.Equal(expected, a.Attributes.Get("window", 0));
        }
    }

    [Fact]
    public void WindowWithLayerListKeepsInterleavedPattern()
    {
        var def = Build(Source(new() { ["decoder.attention.sliding_window"] = 64, ["decoder.attention.sliding_layers"] = new[] { 1, 0 } }));
        var windows = Nodes(def, "CachedAttention").Select(a => a.Attributes.Get("window", 0)).ToList();
        Assert.Equal([64, 0], windows);
    }
}
