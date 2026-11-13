using System.Globalization;
using System.Text;
using Sharplight.Architectures;
using Sharplight.Gguf;
using Sharplight.HuggingFace;
using Sharplight.Models;
using Sharplight.Tensors;
using Sharplight.Tokenizers;
using Xunit;

namespace Sharplight.HuggingFace.Tests;

/// <summary>Opens the HF checkpoint and the llama.cpp-converted GGUF once for all parity tests.</summary>
public sealed class ParityFixture : IDisposable
{
    public static readonly string HfDir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models/hf/Qwen3.5-0.8B");
    public static readonly string GgufPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models/gguf/Qwen3.5-0.8B/Qwen3.5-0.8B-BF16.gguf");

    public bool Available { get; }
    public ModelSource? Gguf { get; }
    /// <summary>HF source without MTP weights: directly comparable to the GGUF (which was converted without them).</summary>
    public ModelSource? Hf { get; }
    /// <summary>HF source with default options (MTP mapped to mtp.*).</summary>
    public ModelSource? HfMtp { get; }

    public ParityFixture()
    {
        Available = Directory.Exists(HfDir) && File.Exists(GgufPath);
        if (!Available) return;
        Gguf = GgufImporter.Open(GgufPath);
        Hf = HuggingFaceImporter.Open(HfDir, new HfImportOptions { IncludeMtp = false });
        HfMtp = HuggingFaceImporter.Open(HfDir);
    }

    public void Dispose()
    {
        foreach (var s in new[] { Gguf, Hf, HfMtp })
            if (s != null) foreach (var r in s.OwnedResources) r.Dispose();
    }
}

public class Qwen35ParityTests(ParityFixture fx) : IClassFixture<ParityFixture>
{
    static void RequireModels(ParityFixture fx) => Assert.SkipUnless(fx.Available, $"model files not found ({ParityFixture.HfDir}, {ParityFixture.GgufPath})");

    [Fact]
    public void Importer_Metadata_Basics()
    {
        RequireModels(fx);
        Assert.Equal("hf", fx.Hf!.Format);
        Assert.Equal("qwen3_5", fx.Hf.Architecture);
        Assert.Equal("qwen3_5", fx.Hf.Get<string>(CanonicalKeys.Architecture));
        Assert.Equal("qwen3_5", fx.Gguf!.Architecture);
        Assert.Equal("qwen3_5", fx.Gguf.Get<string>(CanonicalKeys.Architecture));
        Assert.DoesNotContain(fx.Gguf.Metadata.Keys, k => k.StartsWith("qwen35.", StringComparison.Ordinal));
        Assert.Equal("Qwen3.5-0.8B", fx.Hf.Get<string>(CanonicalKeys.Name));
    }

    [Fact]
    public void Registry_Opens_Directory_And_ShardFile()
    {
        RequireModels(fx);
        var imp = new HuggingFaceImporter();
        Assert.True(imp.CanOpen(ParityFixture.HfDir));
        Assert.True(imp.CanOpen(Directory.EnumerateFiles(ParityFixture.HfDir, "*.safetensors").First()));
        Assert.False(imp.CanOpen(ParityFixture.GgufPath));
        Assert.False(imp.CanOpen(Path.GetTempPath()));
        Assert.False(new GgufImporter().CanOpen(ParityFixture.HfDir));
    }

    [Fact]
    public void Weight_Name_Sets_Equal()
    {
        RequireModels(fx);
        var g = fx.Gguf!.Weights.Keys.ToHashSet();
        var h = fx.Hf!.Weights.Keys.ToHashSet();
        var onlyG = g.Except(h).Order().ToList();
        var onlyH = h.Except(g).Order().ToList();
        Assert.True(onlyG.Count == 0 && onlyH.Count == 0,
            $"only in GGUF ({onlyG.Count}): {string.Join(", ", onlyG)}\nonly in HF ({onlyH.Count}): {string.Join(", ", onlyH)}");
    }

    [Fact]
    public void Weight_Shapes_And_Values_Match()
    {
        RequireModels(fx);
        var problems = new List<string>();
        var report = new SortedDictionary<string, (double MaxAbs, double MaxRel, int Count, bool BitExact)>();
        foreach (var (name, gw) in fx.Gguf!.Weights.OrderBy(p => p.Key, StringComparer.Ordinal))
        {
            if (!fx.Hf!.Weights.TryGetValue(name, out var hw)) continue;
            if (!gw.Shape.Equals(hw.Shape)) { problems.Add($"{name}: shape GGUF {gw.Shape} vs HF {hw.Shape}"); continue; }
            var (maxAbs, maxRel, exact) = Compare(gw, hw);
            // class = name with layer index stripped
            string cls = System.Text.RegularExpressions.Regex.Replace(name, @"layers\.\d+\.", "layers.N.") + $" [{gw.Type}<-{hw.Type}]";
            report.TryGetValue(cls, out var cur);
            report[cls] = (Math.Max(cur.MaxAbs, maxAbs), Math.Max(cur.MaxRel, maxRel), cur.Count + 1, (cur.Count == 0 || cur.BitExact) && exact);
            if (gw.Type != hw.Type) problems.Add($"{name}: type GGUF {gw.Type} vs HF {hw.Type}");
            if (maxRel > 1e-6) problems.Add($"{name}: max rel error {maxRel:g3} (abs {maxAbs:g3})");
        }
        var sb = new StringBuilder("\nper-class comparison (GGUF vs HF, decoded to f32):\n");
        foreach (var (k, v) in report)
            sb.AppendLine($"  {k,-60} n={v.Count,3} bitExact={v.BitExact,-5} maxAbs={v.MaxAbs:g3} maxRel={v.MaxRel:g3}");
        Console.WriteLine(sb);
        Assert.True(problems.Count == 0, string.Join("\n", problems.Take(30)) + sb);
    }

    [Fact]
    public void Untransformed_Weights_Are_ZeroCopy_Mmap_Views()
    {
        RequireModels(fx);
        foreach (var name in new[] { "token_embedding.weight", "layers.0.linear_attention.out.weight", "layers.3.attention.q.weight", "layers.5.mlp.down.weight" })
        {
            var w = fx.Hf!.Weights[name];
            Assert.NotEqual(0, w.Source.TryGetPointer());
            Assert.Equal(DataType.BF16, w.Type);
        }
        // transformed ones are lazy (no pointer)
        Assert.Equal(0, fx.Hf!.Weights["layers.0.input_norm.weight"].Source.TryGetPointer());
        Assert.Equal(0, fx.Hf.Weights["layers.0.linear_attention.a"].Source.TryGetPointer());
    }

    static readonly string[] AdapterKeys =
    [
        "layer_count", "mtp.layer_count", "hidden_size", "ffn_hidden_size", "attention.head_count", "attention.kv_head_count",
        "attention.head_dim", "norm_eps", "rope.dim", "rope.base", "rope.sections", "linear_attention.conv_kernel",
        "linear_attention.key_head_dim", "linear_attention.key_head_count", "linear_attention.value_head_count",
        "linear_attention.value_head_dim", "full_attention_interval", "context_length",
    ];

    [Fact]
    public void Adapter_Hyperparameters_Match()
    {
        RequireModels(fx);
        var problems = new List<string>();
        foreach (var k in AdapterKeys)
        {
            string key = "qwen3_5." + k;
            bool inG = fx.Gguf!.Has(key), inH = fx.Hf!.Has(key);
            if (!inG && !inH) continue;
            if (inG != inH) { problems.Add($"{key}: GGUF present={inG}, HF present={inH}"); continue; }
            if (!ValueEquals(fx.Gguf.Metadata[key], fx.Hf.Metadata[key])) problems.Add($"{key}: GGUF {Fmt(fx.Gguf.Metadata[key])} vs HF {Fmt(fx.Hf.Metadata[key])}");
        }
        Assert.True(problems.Count == 0, string.Join("\n", problems));
    }

    [Fact]
    public void Tokenizer_Metadata_Matches()
    {
        RequireModels(fx);
        var g = fx.Gguf!.Metadata.Where(p => p.Key.StartsWith("tokenizer.", StringComparison.Ordinal)).ToDictionary(p => p.Key, p => p.Value);
        var h = fx.Hf!.Metadata.Where(p => p.Key.StartsWith("tokenizer.", StringComparison.Ordinal)).ToDictionary(p => p.Key, p => p.Value);
        var problems = new List<string>();
        var notes = new List<string>();
        foreach (var key in g.Keys.Union(h.Keys).Order(StringComparer.Ordinal))
        {
            bool inG = g.TryGetValue(key, out var gv), inH = h.TryGetValue(key, out var hv);
            if (inG && inH)
            {
                if (ValueEquals(gv!, hv!)) continue;
                if (key == CanonicalKeys.ChatTemplate) { notes.Add($"{key}: differs (GGUF {((string)gv!).Length} chars vs HF {((string)hv!).Length} chars) - the GGUF comes from a re-packaged template"); continue; }
                if (key == CanonicalKeys.TokenizerPadId)
                {
                    var toks = (string[])g[CanonicalKeys.TokenizerTokens];
                    notes.Add($"{key}: GGUF {Fmt(gv!)}='{toks[Convert.ToInt32(gv)]}' vs HF {Fmt(hv!)}='{toks[Convert.ToInt32(hv)]}' (pad_token differs between the repackaged GGUF and the HF tokenizer_config.json)");
                    continue;
                }
                problems.Add($"{key}: values differ ({Fmt(gv!)} vs {Fmt(hv!)}){ArrayDiff(gv!, hv!)}");
            }
            else if (inH && key is CanonicalKeys.TokenizerAddBos or CanonicalKeys.TokenizerAddEos && hv is false)
                notes.Add($"{key}: HF states false explicitly (llama.cpp's converter does too), GGUF omits it (= preset default, false)");
            else problems.Add($"{key}: only in {(inG ? "GGUF" : "HF")}");
        }
        Console.WriteLine("tokenizer notes:\n  " + string.Join("\n  ", notes));
        Assert.True(problems.Count == 0, string.Join("\n", problems));
    }

    static readonly string[] Samples =
    [
        "Hello, world!", "", " ", "  leading and trailing  ", "The quick brown fox jumps over the lazy dog.",
        "<|im_start|>user\nWrite a haiku.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "<|endoftext|>text<|vision_start|><|image_pad|><|vision_end|>", "<tool_call>{\"a\": 1}</tool_call> <think>x</think>",
        "Unicode: héllo wörld, 你好世界, こんにちは, 😀🚀, Ελληνικά, עברית, ́combining é (é)",
        "numbers 1234567890 3.14159 1e-9 0x7FFF",
        "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)\n\n\tprint(fib(10))  # tab\r\n",
        "int main(void) { printf(\"%d\\n\", 42); return 0; }", "I've been told he's there, 'RE you sure? 'M not sure I'll make it",
        "\n \n\n \n\n\n \t \t\t \t\n  \n   \n    \n     \n🚀 (normal) 😶‍🌫️ (multiple emojis) ✅ 🦙🦙 3 33 333 3333",
        "កាន់តែពិសេសអាច😁 ?我想在apple工作1314151天～ ------======= нещо на Български",
    ];

    [Fact]
    public void Tokenizers_Encode_Identically()
    {
        RequireModels(fx);
        var tg = TokenizerFactory.Create(TokenizerData.FromMetadata(fx.Gguf!.Metadata));
        var th = TokenizerFactory.Create(TokenizerData.FromMetadata(fx.Hf!.Metadata));
        Assert.Equal(tg.VocabSize, th.VocabSize);
        Assert.Equal(tg.BosId, th.BosId);
        Assert.Equal(tg.EosId, th.EosId);
        Assert.Equal(tg.AddBos, th.AddBos);
        Assert.Equal(tg.AddEos, th.AddEos);
        Assert.Equal(tg.EndOfGenerationIds.Order(), th.EndOfGenerationIds.Order());
        for (int i = 0; i < tg.VocabSize; i++)
        {
            Assert.Equal(tg.GetTokenType(i), th.GetTokenType(i));
            Assert.Equal(tg.TokenString(i), th.TokenString(i));
        }
        foreach (var s in Samples)
            foreach (var (addSpecial, parseSpecial) in new[] { (true, true), (false, true), (false, false) })
            {
                var a = tg.Encode(s, addSpecial, parseSpecial);
                var b = th.Encode(s, addSpecial, parseSpecial);
                Assert.True(a.AsSpan().SequenceEqual(b), $"encode mismatch for '{s}' (addSpecial={addSpecial}, parseSpecial={parseSpecial})");
                Assert.Equal(tg.Decode(a), th.Decode(b));
            }
        // sanity: the special tokens really are single ids
        Assert.Single(th.Encode("<|im_start|>", false, true));
    }

    [Fact]
    public void Adapter_Builds_Same_Graphs()
    {
        RequireModels(fx);
        var adapter = new Qwen35Adapter();
        var dg = adapter.Build(new ModelSourceSet { Primary = fx.Gguf! }, new AdapterOptions());
        var dh = ModelAdapterRegistryBuild(fx.Hf!);
        Assert.Equal(dg.Graphs.Keys.Order(), dh.Graphs.Keys.Order());
        foreach (var (name, graph) in dg.Graphs)
            Assert.Equal(graph.Nodes.Count, dh.Graphs[name].Nodes.Count);
        Assert.Equal(dg.States.Count, dh.States.Count);
        Assert.Equal(dg.Property<int>(WellKnown.VocabSize), dh.Property<int>(WellKnown.VocabSize));
        Assert.Equal(dg.Property<int>("n_layer"), dh.Property<int>("n_layer"));
    }

    [Fact]
    public void Mtp_Included_By_Default_As_Mtp_Weights()
    {
        RequireModels(fx);
        var s = fx.HfMtp!;
        Assert.Equal(24, Convert.ToInt32(s.Metadata["qwen3_5.layer_count"]));
        Assert.Equal(1, Convert.ToInt32(s.Metadata["qwen3_5.mtp.layer_count"]));
        var extra = s.Weights.Keys.Except(fx.Hf!.Weights.Keys).Order().ToList();
        string[] expected =
        [
            "mtp.layers.0.attention.k.weight", "mtp.layers.0.attention.k_norm.weight", "mtp.layers.0.input_norm.weight", "mtp.layers.0.attention.o.weight",
            "mtp.layers.0.attention.q.weight", "mtp.layers.0.attention.q_norm.weight", "mtp.layers.0.attention.v.weight", "mtp.layers.0.mlp.down.weight",
            "mtp.layers.0.mlp.gate.weight", "mtp.layers.0.mlp.up.weight", "mtp.fc.weight", "mtp.embedding_norm.weight", "mtp.hidden_norm.weight",
            "mtp.output_norm.weight", "mtp.layers.0.post_attention_norm.weight",
        ];
        Assert.Equal(expected.Order(StringComparer.Ordinal), extra.Order(StringComparer.Ordinal));
        Assert.Equal(new Shape(1024, 2048), s.Weights["mtp.fc.weight"].Shape);
        Assert.Equal(DataType.F32, s.Weights["mtp.embedding_norm.weight"].Type);
        // trunk is unaffected and the adapter reports the MTP layer separately
        var d = ModelAdapterRegistryBuild(s);
        Assert.Equal(24, d.Property<int>("n_layer"));
        Assert.Equal(1, d.Property<int>("mtp_layers"));
        var plain = ModelAdapterRegistryBuild(fx.Hf!);
        Assert.Equal(plain.Graphs[WellKnown.DecodeGraph].Nodes.Count, d.Graphs[WellKnown.DecodeGraph].Nodes.Count);
    }

    [Fact]
    public void ReorderVHeads_Matches_Reference_Permutation()
    {
        // [numK=2, vPerK=3, d=2] -> [vPerK, numK, d]; elements are int16 ids
        int numK = 2, vPerK = 3, d = 2;
        var src = new byte[numK * vPerK * d * 2];
        for (int i = 0; i < src.Length / 2; i++) BitConverter.TryWriteBytes(src.AsSpan(i * 2), (short)i);
        var dst = HfTransforms.ReorderVHeads(src, 2, 1, numK, vPerK, d);
        for (int k = 0; k < numK; k++)
            for (int v = 0; v < vPerK; v++)
                for (int e = 0; e < d; e++)
                    Assert.Equal((short)((k * vPerK + v) * d + e), BitConverter.ToInt16(dst, ((v * numK + k) * d + e) * 2));
        // with an outer dimension (column reorder of a [3, K*V*d] matrix)
        var big = new byte[3 * src.Length];
        for (int i = 0; i < big.Length / 2; i++) BitConverter.TryWriteBytes(big.AsSpan(i * 2), (short)i);
        var rb = HfTransforms.ReorderVHeads(big, 2, 3, numK, vPerK, d);
        int row = numK * vPerK * d;
        for (int r = 0; r < 3; r++)
            for (int k = 0; k < numK; k++)
                for (int v = 0; v < vPerK; v++)
                    Assert.Equal(BitConverter.ToInt16(big, (r * row + (k * vPerK + v) * d) * 2), BitConverter.ToInt16(rb, (r * row + (v * numK + k) * d) * 2));
    }

    // ---- helpers ------------------------------------------------------------------------------

    static ModelDefinition ModelAdapterRegistryBuild(ModelSource src)
    {
        ModelAdapterRegistry.Register(new Qwen35Adapter());
        return ModelAdapterRegistry.Build(new ModelSourceSet { Primary = src });
    }

    static string Fmt(object v) => v is Array a ? $"[{string.Join(",", a.Cast<object>().Take(8))}{(a.Length > 8 ? ",..." : "")}] (len {a.Length})" : v is string s && s.Length > 60 ? s[..60] + "..." : v.ToString() ?? "";

    static string ArrayDiff(object a, object b)
    {
        if (a is not Array x || b is not Array y) return "";
        var diffs = new List<string>();
        int total = 0;
        for (int i = 0; i < Math.Min(x.Length, y.Length); i++)
            if (!ValueEquals(x.GetValue(i)!, y.GetValue(i)!)) { total++; if (diffs.Count < 12) diffs.Add($"[{i}] {Fmt(x.GetValue(i)!)} vs {Fmt(y.GetValue(i)!)}"); }
        return $"\n    {total} differing elements: {string.Join("; ", diffs)}";
    }

    static bool ValueEquals(object a, object b)
    {
        if (a is string sa && b is string sb) return sa == sb;
        if (a is Array aa && b is Array ba)
        {
            if (aa.Length != ba.Length) return false;
            for (int i = 0; i < aa.Length; i++) if (!ValueEquals(aa.GetValue(i)!, ba.GetValue(i)!)) return false;
            return true;
        }
        if (a is bool x && b is bool y) return x == y;
        try
        {
            double da = Convert.ToDouble(a, CultureInfo.InvariantCulture), db = Convert.ToDouble(b, CultureInfo.InvariantCulture);
            return da == db || Math.Abs(da - db) <= 1e-9 * Math.Max(Math.Abs(da), Math.Abs(db));
        }
        catch { return false; }
    }

    /// <summary>Compares two weights chunk-wise after decoding to f32. Returns (max abs error, max relative error, bit-exact bytes).</summary>
    static (double MaxAbs, double MaxRel, bool BitExact) Compare(WeightInfo a, WeightInfo b)
    {
        long n = a.Shape.Elements();
        const int Chunk = 1 << 20;
        double maxAbs = 0, maxRel = 0;
        bool exact = a.Type == b.Type;
        for (long start = 0; start < n; start += Chunk)
        {
            int count = (int)Math.Min(Chunk, n - start);
            var ba = ReadChunk(a, start, count);
            var bb = ReadChunk(b, start, count);
            if (exact && !ba.AsSpan().SequenceEqual(bb)) exact = false;
            if (a.Type == b.Type && ba.AsSpan().SequenceEqual(bb)) continue;
            var fa = ToF32(a.Type, ba, count);
            var fb = ToF32(b.Type, bb, count);
            for (int i = 0; i < count; i++)
            {
                double abs = Math.Abs((double)fa[i] - fb[i]);
                double rel = abs / Math.Max(Math.Max(Math.Abs((double)fa[i]), Math.Abs((double)fb[i])), 1e-30);
                if (abs > maxAbs) maxAbs = abs;
                if (rel > maxRel) maxRel = rel;
            }
        }
        return (maxAbs, maxRel, exact);
    }

    static byte[] ReadChunk(WeightInfo w, long startElem, int count)
    {
        int es = w.Type.ElementSize;
        var buf = new byte[count * es];
        w.Source.Read(startElem * es, buf);
        return buf;
    }

    static float[] ToF32(DataType t, byte[] bytes, int count)
    {
        var f = new float[count];
        if (t == DataType.F32) for (int i = 0; i < count; i++) f[i] = BitConverter.ToSingle(bytes, i * 4);
        else if (t == DataType.BF16) for (int i = 0; i < count; i++) f[i] = BitConverter.Int32BitsToSingle(BitConverter.ToUInt16(bytes, i * 2) << 16);
        else if (t == DataType.F16) for (int i = 0; i < count; i++) f[i] = (float)BitConverter.ToHalf(bytes, i * 2);
        else throw new NotSupportedException(t.ToString());
        return f;
    }
}
