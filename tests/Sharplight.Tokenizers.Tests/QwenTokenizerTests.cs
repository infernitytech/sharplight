using System.Diagnostics;
using System.Text;
using Sharplight.Tokenizers.Bpe;
using Xunit;

namespace Sharplight.Tokenizers.Tests;

/// <summary>Tests against the real Qwen3.8 tokenizer; skipped unless .work/tokenizer/qwen35.json exists.</summary>
public class QwenTokenizerTests
{
    private static ITokenizer Tok()
    {
        Assert.SkipUnless(TestData.QwenTokenizer() != null, "qwen35.json not exported (run .work/tokenizer/export_gguf_tokenizer.py)");
        return TestData.QwenTokenizer()!;
    }

    [Fact]
    public void SpecialTokenIdsAndEndOfGeneration()
    {
        var t = Tok();
        Assert.Equal(248320, t.VocabSize);
        Assert.Equal(248045, t.TokenId("<|im_start|>"));
        Assert.Equal(248046, t.TokenId("<|im_end|>"));
        Assert.Equal(248044, t.TokenId("<|endoftext|>"));
        Assert.Equal(248068, t.TokenId("<think>"));
        Assert.Equal(248069, t.TokenId("</think>"));
        Assert.Equal(248046, t.EosId);
        Assert.False(t.AddBos);
        Assert.Equal([248044, 248046], t.EndOfGenerationIds);
        Assert.True(t.IsEndOfGeneration(248046));
        Assert.False(t.IsEndOfGeneration(248045));
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void EncodeMatchesLlamaCpp(int index)
    {
        var t = Tok();
        var c = TestData.Fixture()[index];
        Assert.Equal(c.Ids, t.Encode(c.Text, addSpecial: true, parseSpecial: c.ParseSpecial));
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void DecodeRoundTrips(int index)
    {
        var t = Tok();
        var c = TestData.Fixture()[index];
        string expected = c.Text;
        // control-token text is rendered literally on decode, so parse-special and plain text both round-trip
        Assert.Equal(expected, t.Decode(t.Encode(c.Text, parseSpecial: c.ParseSpecial)));
        Assert.Equal(expected, t.Decode(c.Ids));
    }

    public static IEnumerable<object[]> Cases() => TestData.FixtureIndexes();

    [Fact]
    public void StreamingDecoderReproducesDecodeForEveryFixtureCase()
    {
        var t = Tok();
        foreach (var c in TestData.Fixture())
        {
            var dec = new StreamingDecoder(t);
            var sb = new StringBuilder();
            foreach (int id in c.Ids)
            {
                string piece = dec.Push(id);
                if (c.Text.Contains('�') == false)
                {
                    Assert.DoesNotContain('�', piece);
                }

                sb.Append(piece);
            }
            sb.Append(dec.Flush());
            Assert.Equal(c.Text, sb.ToString());
        }
    }

    [Fact]
    public void StreamingDecoderSplitsEmojiAcrossTokens()
    {
        var t = Tok();
        // 4-byte emoji and CJK typically come as several byte-level tokens; the streamed pieces must be whole characters
        string text = "Smile 😀 and 日本語 and 🇩🇪";
        int[] ids = t.Encode(text);
        var dec = new StreamingDecoder(t);
        var pieces = ids.Select(dec.Push).ToList();
        Assert.Equal(text, string.Concat(pieces) + dec.Flush());
        Assert.Contains("", pieces);                     // at least one token carried only a partial character
        Assert.All(pieces, p => Assert.DoesNotContain('�', p));
        // force splitting at byte level
        var dec2 = new StreamingDecoder(t);
        var sb = new StringBuilder();
        foreach (byte b in Encoding.UTF8.GetBytes(text))
        {
            sb.Append(dec2.Push(t.TokenId(ByteEncoding.ByteToChar[b].ToString())!.Value));
        }

        Assert.Equal(text, sb.ToString());
    }

    [Fact]
    public void ChatTemplateTokens()
    {
        var t = Tok();
        int[] ids = t.Encode("<|im_start|>user\nHi<|im_end|>\n");
        Assert.Equal(248045, ids[0]);
        Assert.Equal("user", t.Decode(ids[1..2]));
        Assert.Contains(248046, ids);
        Assert.Equal("user\nHi\n", t.Decode(ids, skipSpecial: true));
    }

    [Fact]
    public void UnusedTokensDecodeToNothing()
        => Assert.Equal("", Tok().Decode([248100]));

    [Fact]
    public void HeapPathMatchesScanPathOnRealVocab()
    {
        Tok();
        var data = TestData.Qwen35()!;
        var scan = new ByteLevelBpeTokenizer(data) { HeapThreshold = int.MaxValue };
        var heap = new ByteLevelBpeTokenizer(data) { HeapThreshold = 0 };
        foreach (var c in TestData.Fixture().Take(80))
        {
            Assert.Equal(scan.Encode(c.Text, true, c.ParseSpecial), heap.Encode(c.Text, true, c.ParseSpecial));
        }

        string wordy = string.Concat(Enumerable.Range(0, 4000).Select(i => (char)('a' + (i * 7 % 26))));
        Assert.Equal(scan.Encode(wordy), heap.Encode(wordy));
        Assert.Equal(wordy, scan.Decode(scan.Encode(wordy)));
    }

    [Theory]
    [InlineData("default")]
    [InlineData("gpt-2")]
    [InlineData("llama-bpe")]
    [InlineData("qwen2")]
    [InlineData("qwen35")]
    public void EveryPresetRoundTrips(string pre)
    {
        Tok();
        var t = TokenizerFactory.Create(TestData.Qwen35()! with { PreTokenizer = pre });
        foreach (var c in TestData.Fixture())
        {
            Assert.Equal(c.Text, t.Decode(t.Encode(c.Text, addSpecial: false, parseSpecial: c.ParseSpecial)));
        }
    }

    [Fact]
    public void Qwen2AndQwen35DifferOnCombiningMarks()
    {
        // "ab" + combining acute + "cd": qwen35 keeps the mark inside the letter run, qwen2 starts a new word at the mark
        int[] cpts = [(int)'a', 'b', 0x301, 'c', 'd'];
        var chars = new int[cpts.Length + 1];
        var q35 = new List<int>();
        var q2 = new List<int>();
        new RegexFamilySplitter(RegexFamily.Llama, true, 1, marks: true).Split(new SplitInput("", cpts, chars), q35);
        new RegexFamilySplitter(RegexFamily.Llama, true, 1, marks: false).Split(new SplitInput("", cpts, chars), q2);
        Assert.Equal([5], q35);
        Assert.Equal([2, 3], q2);
    }

    /// <summary>Sweep of every Unicode code point plus real text and random mixtures, generated by .work/tokenizer/gen_fixture.py.</summary>
    [Fact]
    public void SweepMatchesLlamaCpp()
    {
        var t = Tok();
        string? path = TestData.FindWorkFile("sweep.json");
        Assert.SkipUnless(path != null, "sweep.json not generated");
        var cases = TestData.LoadCases(path!);
        var bad = new List<string>();
        for (int i = 0; i < cases.Count; i++)
        {
            int[] got = t.Encode(cases[i].Text, true, cases[i].ParseSpecial);
            if (!got.AsSpan().SequenceEqual(cases[i].Ids))
            {
                int k = 0;
                while (k < got.Length && k < cases[i].Ids.Length && got[k] == cases[i].Ids[k])
                {
                    k++;
                }

                bad.Add($"case {i}: first difference at token {k} (got {(k < got.Length ? got[k] : -1)}, expected {(k < cases[i].Ids.Length ? cases[i].Ids[k] : -1)}), context: {Context(t, cases[i].Ids, k)}");
            }
        }
        Assert.True(bad.Count == 0, $"{bad.Count}/{cases.Count} cases differ:\n" + string.Join("\n", bad.Take(20)));
    }

    private static string Context(ITokenizer t, int[] ids, int k)
        => System.Text.Json.JsonSerializer.Serialize(t.Decode(ids.AsSpan(Math.Max(0, k - 3), Math.Min(8, ids.Length - Math.Max(0, k - 3)))));

    [Fact]
    public void EncodeThroughput()
    {
        Tok();
        // ~1 MB of mixed real text; a fresh tokenizer so the word cache starts cold
        var sb = new StringBuilder();
        string? root = TestData.FindWorkFile("qwen35.json") is { } p ? Path.GetFullPath(Path.Combine(Path.GetDirectoryName(p)!, "..")) : null;
        if (root != null && Directory.Exists(Path.Combine(root, "llama.cpp")))
        {
            foreach (string f in Directory.EnumerateFiles(Path.Combine(root, "llama.cpp", "docs"), "*.md", SearchOption.AllDirectories)
                .Concat(Directory.EnumerateFiles(Path.Combine(root, "llama.cpp", "src"), "*.cpp")).OrderBy(x => x, StringComparer.Ordinal))
            {
                sb.Append(File.ReadAllText(f));
                if (sb.Length >= 1_000_000)
                {
                    break;
                }
            }
        }
        foreach (var c in TestData.Fixture())
        {
            if (sb.Length >= 1_100_000)
            {
                break;
            }
            sb.Append(c.Text).Append('\n');
        }
        string text = sb.ToString();
        int bytes = Encoding.UTF8.GetByteCount(text);

        var tok = TokenizerFactory.Create(TestData.Qwen35()!);
        tok.Encode("warm up");
        var sw = Stopwatch.StartNew();
        int[] cold = tok.Encode(text);
        double coldS = sw.Elapsed.TotalSeconds;
        // second fresh instance: empty word cache but JIT already warm (steady-state cold-cache speed)
        var tok2 = TokenizerFactory.Create(TestData.Qwen35()!);
        sw.Restart();
        tok2.Encode(text);
        double cold2S = sw.Elapsed.TotalSeconds;
        coldS = Math.Min(coldS, cold2S);
        sw.Restart();
        int[] warm = tok.Encode(text);
        double warmS = sw.Elapsed.TotalSeconds;
        Assert.Equal(cold, warm);

        string line = $"text={bytes / 1e6:F2} MB tokens={cold.Length} cold={bytes / 1e6 / coldS:F1} MB/s ({cold.Length / coldS / 1e3:F0}k tok/s) warm-cache={bytes / 1e6 / warmS:F1} MB/s";
        TestContext.Current.SendDiagnosticMessage(line);
        if (TestData.FindWorkFile("qwen35.json") is { } q)
        {
            File.WriteAllText(Path.Combine(Path.GetDirectoryName(q)!, "throughput.txt"), line + "\n");
        }

        Assert.True(bytes / 1e6 / coldS > 0.5, line);
    }
}
