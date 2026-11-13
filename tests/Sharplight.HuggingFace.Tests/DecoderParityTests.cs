using System.Buffers.Binary;
using System.Globalization;
using System.Text;
using Sharplight.Gguf;
using Sharplight.Models;
using Sharplight.Tensors;
using Sharplight.Tokenizers;
using Xunit;

namespace Sharplight.HuggingFace.Tests;

/// <summary>
/// The Hugging Face and the llama.cpp-converted GGUF checkpoint of the same decoder-only model must import to the same
/// canonical description (<c>decoder.*</c> hyper-parameters, weight names, shapes and values up to quantization, tokenizer).
/// Cases skip when the model files are not present under ~/models.
/// </summary>
public class DecoderParityTests
{
    private static readonly string Home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);

    public static IEnumerable<TheoryDataRow<string, string, string>> Models() =>
    [
        new("Llama-3.2-1B-Instruct", "hf/Llama-3.2-1B-Instruct", "gguf/Llama-3.2-1B-Instruct/Llama-3.2-1B-Instruct-Q8_0.gguf"),
        new("Qwen2.5-0.5B-Instruct", "hf/Qwen2.5-0.5B-Instruct", "gguf/Qwen2.5-0.5B-Instruct/qwen2.5-0.5b-instruct-q8_0.gguf"),
        new("Qwen2.5-0.5B-Instruct-f16", "hf/Qwen2.5-0.5B-Instruct", "gguf/Qwen2.5-0.5B-Instruct/qwen2.5-0.5b-instruct-fp16.gguf"),
        new("Qwen3-0.6B", "hf/Qwen3-0.6B", "gguf/Qwen3-0.6B/Qwen3-0.6B-Q8_0.gguf"),
        new("SmolLM2-360M-Instruct", "hf/SmolLM2-360M-Instruct", "gguf/SmolLM2-360M-Instruct/smollm2-360m-instruct-q8_0.gguf"),
        new("Phi-3.5-mini-instruct", "hf/Phi-3.5-mini-instruct", "gguf/Phi-3.5-mini-instruct/Phi-3.5-mini-instruct-Q4_K_M.gguf"),
        new("OLMoE-1B-7B-0924-Instruct", "hf/OLMoE-1B-7B-0924-Instruct", "gguf/OLMoE-1B-7B-0924-Instruct/OLMoE-1B-7B-0924-Instruct-Q4_K_M.gguf"),
        new("gemma-3-1b-it", "hf/gemma-3-1b-it", "gguf/gemma-3-1b-it/gemma-3-1b-it-Q8_0.gguf"),
        new("gemma-2-2b-it", "hf/gemma-2-2b-it", "gguf/gemma-2-2b-it/gemma-2-2b-it-Q8_0.gguf"),
        new("gemma-4-E2B-it-Q8_0", "hf/gemma-4-E2B-it", "gguf/gemma-4-E2B-it/gemma-4-E2B-it-Q8_0.gguf"),
        new("gemma-4-E2B-it-BF16", "hf/gemma-4-E2B-it", "gguf/gemma-4-E2B-it/gemma-4-E2B-it-BF16.gguf"),
        new("gemma-4-12B-it-BF16 (k = v global layers)", "hf/gemma-4-12B-it", "gguf/gemma-4-12b-it/gemma-4-12b-it-BF16.gguf"),   // 23 GB; skipped when the BF16 GGUF is not downloaded
        new("NVIDIA-Nemotron-3-Nano-4B-Q8_0", "hf/NVIDIA-Nemotron-3-Nano-4B", "gguf/NVIDIA-Nemotron-3-Nano-4B/NVIDIA-Nemotron-3-Nano-4B-Q8_0.gguf"),
        new("NVIDIA-Nemotron-3-Nano-4B-BF16", "hf/NVIDIA-Nemotron-3-Nano-4B", "gguf/NVIDIA-Nemotron-3-Nano-4B/NVIDIA-Nemotron-3-Nano-4B-BF16.gguf"),
        new("Muse-Glimmer-30B-Q4_K_XL", "hf/Muse-Glimmer-30B", "gguf/Muse-Glimmer-30B/Muse-Glimmer-30B-UD-Q4_K_XL.gguf"),
        new("Bonsai-1.7B-Q1_0", "hf/Bonsai-1.7B", "gguf/Bonsai-1.7B/Bonsai-1.7B-Q1_0.gguf"),
        new("Ternary-Bonsai-1.7B-PQ2_0", "hf/Ternary-Bonsai-1.7B", "gguf/Ternary-Bonsai-1.7B/Ternary-Bonsai-1.7B-PQ2_0.gguf"),
        new("Ternary-Bonsai-1.7B-Q2_0_g64", "hf/Ternary-Bonsai-1.7B", "gguf/Ternary-Bonsai-1.7B/Ternary-Bonsai-1.7B-Q2_0_g64.gguf"),
    ];

    private static (ModelSource Hf, ModelSource Gguf)? Open(string hf, string gguf)
    {
        string h = Path.Combine(Home, "models", hf), g = Path.Combine(Home, "models", gguf);
        if (!Directory.Exists(h) || !File.Exists(g))
        {
            return null;
        }

        try { return (HuggingFaceImporter.Open(h), GgufImporter.Open(g)); }
        catch (NotSupportedException) { return null; }   // family not supported by this build
    }

    [Theory, MemberData(nameof(Models))]
    public void Hyperparameters_And_WeightNames_Match(string name, string hf, string gguf)
    {
        if (Open(hf, gguf) is not var (h, g)) { Assert.Skip($"{name}: files missing or unsupported"); return; }
        using var _1 = new Disposer(h); using var _2 = new Disposer(g);
        Assert.Equal(g.Architecture, h.Architecture);
        Assert.Contains(g.Architecture, new[] { "decoder", "nemotron_h" });
        var problems = new List<string>();
        foreach (var key in h.Metadata.Keys.Concat(g.Metadata.Keys).Where(k => k.StartsWith(g.Architecture + ".", StringComparison.Ordinal)).Distinct().Order(StringComparer.Ordinal))
        {
            bool inH = h.Metadata.TryGetValue(key, out var hv), inG = g.Metadata.TryGetValue(key, out var gv);
            if (key.EndsWith(".context_length"))
            {
                continue;   // repackagers sometimes truncate; informational
            }

            if (inH != inG)
            {
                problems.Add($"{key}: HF present={inH} GGUF present={inG}");
            }
            else if (!ValueEquals(hv!, gv!))
            {
                problems.Add($"{key}: HF {Fmt(hv!)} vs GGUF {Fmt(gv!)}");
            }
        }
        // A tied head may be stored explicitly by one format (then it equals the embedding): not a canonical difference.
        var onlyH = h.Weights.Keys.Except(g.Weights.Keys).Where(k => k != "lm_head.weight").Order().ToList();
        var onlyG = g.Weights.Keys.Except(h.Weights.Keys).Where(k => k != "lm_head.weight").Order().ToList();
        if (onlyH.Count + onlyG.Count > 0)
        {
            problems.Add($"weights only in HF: {string.Join(", ", onlyH.Take(8))}; only in GGUF: {string.Join(", ", onlyG.Take(8))}");
        }

        Assert.True(problems.Count == 0, string.Join("\n", problems));
    }

    [Theory, MemberData(nameof(Models))]
    public void Weight_Shapes_And_Values_Match(string name, string hf, string gguf)
    {
        if (name.StartsWith("Qwen2.5", StringComparison.Ordinal)) { Assert.Skip("the official Qwen2.5 GGUF repo and HF repo carry different weight revisions of layer 0 (checked against fp16); covered by the generation test"); return; }
        if (Open(hf, gguf) is not var (h, g)) { Assert.Skip($"{name}: files missing or unsupported"); return; }
        using var _1 = new Disposer(h); using var _2 = new Disposer(g);
        var problems = new List<string>();
        var report = new SortedDictionary<string, double>();
        foreach (var (wn, gw) in g.Weights.OrderBy(p => p.Key, StringComparer.Ordinal))
        {
            if (!h.Weights.TryGetValue(wn, out var hw))
            {
                continue;
            }
            // Only layers 0..2 and the last one: all layers share the mapping code.
            var lm = System.Text.RegularExpressions.Regex.Match(wn, @"^layers\.(\d+)\.");
            if (lm.Success && int.Parse(lm.Groups[1].Value) is var il && il > 2 && il != g.Arch<int>("layer_count") - 1)
            {
                continue;
            }

            if (!gw.Shape.Equals(hw.Shape)) { problems.Add($"{wn}: shape GGUF {gw.Shape} vs HF {hw.Shape}"); continue; }
            if (gw.Type != DataType.F32 && gw.Type != DataType.F16 && gw.Type != DataType.BF16 && gw.Type != DataType.Q8_0 && !IsGroupBitFormat(gw.Type))
            {
                continue;   // shape-only for other quantizations
            }

            double err = RelRms(gw, hw);
            string cls = System.Text.RegularExpressions.Regex.Replace(wn, @"layers\.\d+\.", "layers.N.") + $" [{gw.Type}<-{hw.Type}]";
            report[cls] = Math.Max(report.GetValueOrDefault(cls), err);
            double tol = gw.Type.IsQuantized ? 0.03 : 2e-3;
            if (err > tol)
            {
                problems.Add($"{wn}: relative rms error {err:g3} > {tol} ({gw.Type} vs {hw.Type})");
            }
        }
        var sb = new StringBuilder("\n");
        foreach (var (k, v) in report)
        {
            sb.AppendLine($"  {k,-60} relRms={v:g3}");
        }

        Console.WriteLine(name + sb);
        Assert.True(problems.Count == 0, string.Join("\n", problems.Take(20)) + sb);
    }

    [Theory, MemberData(nameof(Models))]
    public void Tokenizers_Agree(string name, string hf, string gguf)
    {
        if (Open(hf, gguf) is not var (h, g)) { Assert.Skip($"{name}: files missing or unsupported"); return; }
        using var _1 = new Disposer(h); using var _2 = new Disposer(g);
        var th = TokenizerFactory.Create(TokenizerData.FromMetadata(h.Metadata));
        var tg = TokenizerFactory.Create(TokenizerData.FromMetadata(g.Metadata));
        Assert.Equal(tg.VocabSize, th.VocabSize);
        Assert.Equal(tg.BosId, th.BosId);
        // The unsloth Nemotron GGUF enables add_bos_token while the official tokenizer_config.json disables it: a repackaging choice, compare plain encodings then.
        bool sameBos = tg.AddBos == th.AddBos;
        if (!name.StartsWith("NVIDIA-Nemotron", StringComparison.Ordinal))
        {
            Assert.True(sameBos, "add_bos differs");
        }

        if (name.Contains("12B", StringComparison.Ordinal))
        {
            Assert.Subset(th.EndOfGenerationIds.ToHashSet(), tg.EndOfGenerationIds.ToHashSet());   // the unsloth 12B GGUF lists only <turn|> (eos id 106), the HF generation_config also <eos>
        }
        else
        {
            Assert.Equal(tg.EndOfGenerationIds.Order(), th.EndOfGenerationIds.Order());
        }

        foreach (var s in new[]
        {
            "Hello, world!", "The quick brown fox jumps over the lazy dog.", "  leading and trailing  ", "numbers 1234567890 3.14159",
            "Unicode: héllo wörld, 你好世界, 😀🚀", "def fib(n):\n    return n if n < 2 else fib(n-1) + fib(n-2)\n\n\tprint(fib(10))",
            "I've been told he's there, 'RE you sure?", "<start_of_turn>user\nHi<end_of_turn>\n<|im_start|>x<|im_end|>",
            "<|turn>user\nHello there<turn|>\n<|turn>model\n", "a\n\n\n\nb   c    d\n \n", "Ünïcödé  mixed 日本語のテキスト and emoji 🎉👍🏽", "x = [1, 2, 3]  # comment\n\treturn x",
        })
        {
            var a = tg.Encode(s, addSpecial: sameBos, parseSpecial: true);
            var b = th.Encode(s, addSpecial: sameBos, parseSpecial: true);
            Assert.True(a.AsSpan().SequenceEqual(b), $"encode mismatch for '{s}': GGUF [{string.Join(",", a)}] vs HF [{string.Join(",", b)}]");
        }
    }

    // ---- helpers ------------------------------------------------------------------------------
    private sealed class Disposer(ModelSource s) : IDisposable
    {
        public void Dispose() { foreach (var r in s.OwnedResources)
            {
                r.Dispose();
            }
        }
    }

    private static string Fmt(object v) => v is Array a ? $"[{string.Join(",", a.Cast<object>().Take(8))}{(a.Length > 8 ? ",..." : "")}] (len {a.Length})" : v.ToString() ?? "";

    private static bool ValueEquals(object a, object b)
    {
        if (a is string sa && b is string sb)
        {
            return sa == sb;
        }

        if (a is Array aa && b is Array ba)
        {
            if (aa.Length != ba.Length)
            {
                return false;
            }

            for (int i = 0; i < aa.Length; i++)
            {
                if (!ValueEquals(aa.GetValue(i)!, ba.GetValue(i)!))
                {
                    return false;
                }
            }

            return true;
        }
        if (a is bool x && b is bool y)
        {
            return x == y;
        }

        try
        {
            double da = Convert.ToDouble(a, CultureInfo.InvariantCulture), db = Convert.ToDouble(b, CultureInfo.InvariantCulture);
            return da == db || Math.Abs(da - db) <= 1e-5 * Math.Max(Math.Abs(da), Math.Abs(db));
        }
        catch { return false; }
    }

    /// <summary>sqrt(Σ(a-b)²/Σa²) over the first up-to-4M elements.</summary>
    private static double RelRms(WeightInfo a, WeightInfo b)
    {
        var fa = Decode(a); var fb = Decode(b);
        double num = 0, den = 0;
        for (int i = 0; i < fa.Length; i++) { double d = fa[i] - fb[i]; num += d * d; den += (double)fa[i] * fa[i]; }
        return den == 0 ? (num == 0 ? 0 : double.PositiveInfinity) : Math.Sqrt(num / den);
    }

    private static bool IsGroupBitFormat(DataType t) => t == DataType.Q1_0 || t == DataType.Q2_0 || t == DataType.Q2_0_G128;

    private static float[] Decode(WeightInfo w)
    {
        long n = Math.Min(w.Shape.Elements(), 4L << 20);
        if (IsGroupBitFormat(w.Type))
        {
            // Bonsai 1-bit / ternary: one f16 scale per group; 1 sign bit or a 2-bit code ((code-1) * scale).
            int bs = w.Type.BlockSize, bb = w.Type.BlockBytes;
            n = n / bs * bs;
            var raw = new byte[n / bs * bb];
            w.Source.Read(0, raw);
            var f = new float[n];
            for (long blk = 0; blk < n / bs; blk++)
            {
                float d = (float)BinaryPrimitives.ReadHalfLittleEndian(raw.AsSpan((int)(blk * bb)));
                for (int i = 0; i < bs; i++)
                {
                    f[blk * bs + i] = w.Type == DataType.Q1_0
                        ? (((raw[blk * bb + 2 + i / 8] >> (i % 8)) & 1) != 0 ? d : -d)
                        : (((raw[blk * bb + 2 + i / 4] >> (2 * (i % 4))) & 3) - 1) * d;
                }
            }
            return f;
        }
        if (w.Type == DataType.Q8_0)
        {
            n = n / 32 * 32;
            var raw = new byte[n / 32 * 34];
            w.Source.Read(0, raw);
            var f = new float[n];
            for (long blk = 0; blk < n / 32; blk++)
            {
                float d = (float)BinaryPrimitives.ReadHalfLittleEndian(raw.AsSpan((int)(blk * 34)));
                for (int i = 0; i < 32; i++)
                {
                    f[blk * 32 + i] = d * (sbyte)raw[blk * 34 + 2 + i];
                }
            }
            return f;
        }
        int es = w.Type.ElementSize;
        var bytes = new byte[n * es];
        w.Source.Read(0, bytes);
        var r = new float[n];
        for (int i = 0; i < n; i++)
        {
            r[i] = w.Type == DataType.F32 ? BitConverter.ToSingle(bytes, i * 4)
                 : w.Type == DataType.BF16 ? BitConverter.Int32BitsToSingle(BitConverter.ToUInt16(bytes, i * 2) << 16)
                 : w.Type == DataType.F16 ? (float)BitConverter.ToHalf(bytes, i * 2)
                 : throw new NotSupportedException(w.Type.ToString());
        }

        return r;
    }
}
