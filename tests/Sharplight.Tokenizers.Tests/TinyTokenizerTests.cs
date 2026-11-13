using Sharplight.Tokenizers.Bpe;
using Xunit;

namespace Sharplight.Tokenizers.Tests;

public class TinyTokenizerTests
{
    [Fact]
    public void MergesApplyInRankOrder()
    {
        var t = TokenizerFactory.Create(TestData.Tiny());
        int[] ids = t.Encode("hello world");
        Assert.Equal(["hello", "Ġworld"], ids.Select(i => t.TokenString(i)));
        Assert.Equal("hello world", t.Decode(ids));
    }

    [Fact]
    public void SpecialTokensAreParsedOnlyWhenRequested()
    {
        var t = TokenizerFactory.Create(TestData.Tiny());
        int s = t.TokenId("<s>")!.Value, u = t.TokenId("<usr>")!.Value;
        Assert.Equal([s, .. t.Encode("hello"), u], t.Encode("<s>hello<usr>"));
        // control token is plain text without parseSpecial; user-defined tokens are still matched (llama.cpp behaviour)
        int[] plain = t.Encode("<s>x<usr>", parseSpecial: false);
        Assert.DoesNotContain(s, plain);
        Assert.Contains(u, plain);
        Assert.Equal("<s>x<usr>", t.Decode(plain));
    }

    [Fact]
    public void SkipSpecialHidesControlTokensOnly()
    {
        var t = TokenizerFactory.Create(TestData.Tiny());
        int[] ids = t.Encode("<s>a<usr>b</s>");
        Assert.Equal("<s>a<usr>b</s>", t.Decode(ids));
        Assert.Equal("a<usr>b", t.Decode(ids, skipSpecial: true));
    }

    [Fact]
    public void AddBosAndEos()
    {
        var d = TestData.Tiny(addBos: true) with { AddEos = true };
        var t = TokenizerFactory.Create(d);
        int[] ids = t.Encode("hello");
        Assert.Equal(t.BosId, ids[0]);
        Assert.Equal(t.EosId, ids[^1]);
        Assert.Equal(1, t.Encode("hello", addSpecial: false).Length);
        Assert.Equal([t.EosId!.Value], t.EndOfGenerationIds);
        Assert.True(t.IsEndOfGeneration(t.EosId.Value));
    }

    [Fact]
    public void EmptyTextAndNonVocabBytes()
    {
        var t = TokenizerFactory.Create(TestData.Tiny());
        Assert.Empty(t.Encode(""));
        Assert.Equal("", t.Decode([]));
        Assert.Equal("日本語😀", t.Decode(t.Encode("日本語😀")));
    }

    [Fact]
    public void PriorityQueuePathMatchesScanPath()
    {
        var data = TestData.Tiny();
        var scan = new ByteLevelBpeTokenizer(data) { HeapThreshold = int.MaxValue };
        var heap = new ByteLevelBpeTokenizer(data) { HeapThreshold = 0 };
        string text = string.Concat(Enumerable.Repeat("hello world helloworld lllld ", 200));
        Assert.Equal(scan.Encode(text), heap.Encode(text));
        string longWord = string.Concat(Enumerable.Repeat("hellowor", 700));
        Assert.Equal(scan.Encode(longWord), heap.Encode(longWord));
    }

    [Fact]
    public void MergeResultsOutsideVocabularyFallBackToBytes()
    {
        // "a b" -> "ab" is not in the vocab; llama.cpp still merges by string, then falls back to single bytes
        var tokens = Enumerable.Range(0, 256).Select(b => ByteEncoding.ByteToChar[b].ToString()).ToArray();
        var t = TokenizerFactory.Create(new TokenizerData { Model = "gpt2", Tokens = tokens, Merges = ["a b"] });
        Assert.Equal(["a", "b"], t.Encode("ab").Select(i => t.TokenString(i)));
        Assert.Equal("ab", t.Decode(t.Encode("ab")));
    }
}
