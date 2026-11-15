using Sharplight.Tokenizers.Spm;
using Xunit;

namespace Sharplight.Tokenizers.Tests;

/// <summary>Synthetic-vocabulary tests of the SentencePiece-style ("spm") tokenizer.</summary>
public class SentencePieceTests
{
    // ids: 0 <unk>, 1 <s>, 2 </s>, 3.. pieces, bytes <0x00>..<0xFF> at the end
    private static SentencePieceTokenizer Make(bool addSpacePrefix = true)
    {
        var tokens = new List<string> { "<unk>", "<s>", "</s>", "▁", "a", "b", "c", "ab", "abc", "▁a", "▁ab", "▁abc", "<end_of_turn>" };
        var scores = new List<float> { 0, 0, 0, -1, -2, -2, -2, -1.5f, -1.2f, -1.4f, -1.1f, -0.5f, 0 };
        var types = new List<int> { 2, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4 };
        for (int i = 0; i < 256; i++)
        {
            tokens.Add($"<0x{i:X2}>");
            scores.Add(0);
            types.Add(6);
        }
        return new SentencePieceTokenizer(new TokenizerData
        {
            Model = "spm",
            Tokens = [.. tokens],
            Scores = [.. scores],
            TokenTypes = [.. types],
            BosId = 1,
            EosId = 2,
            UnkId = 0,
            AddBos = true,
            AddSpacePrefix = addSpacePrefix,
        });
    }

    [Fact]
    public void MergesBestScoreFirstAndPrefixesSpace()
    {
        var t = Make();
        // "abc" -> "▁abc" (dummy prefix) = id 11 after merging through ▁a (-1.4) / ab (-1.5) / ▁ab (-1.1) / ▁abc (-0.5)
        Assert.Equal([1, 11], t.Encode("abc"));
        Assert.Equal([11], t.Encode("abc", addSpecial: false));
        Assert.Equal(" abc", t.Decode([1, 11], skipSpecial: true));   // llama.cpp: a skipped BOS keeps the dummy-prefix space
        Assert.Equal("abc", t.Decode([11]));
    }

    [Fact]
    public void NoSpacePrefixWhenDisabled()
    {
        var t = Make(addSpacePrefix: false);
        Assert.Equal([8], t.Encode("abc", addSpecial: false));   // "abc" token
    }

    [Fact]
    public void UnknownCharactersFallBackToBytes()
    {
        var t = Make();
        var ids = t.Encode("é", addSpecial: false);            // "▁" + C3 A9 as byte tokens
        Assert.Equal(3, ids[0]);
        Assert.Equal(t.TokenId("<0xC3>"), ids[1]);
        Assert.Equal(t.TokenId("<0xA9>"), ids[2]);
        Assert.Equal("é", t.Decode(ids));
    }

    [Fact]
    public void SpecialTokensAreParsedOnlyWhenRequested()
    {
        var t = Make();
        Assert.Contains(12, t.Encode("a<end_of_turn>b", addSpecial: false, parseSpecial: true));
        Assert.Contains(t.TokenId("<end_of_turn>")!.Value, t.Encode("a<end_of_turn>", addSpecial: false, parseSpecial: false));   // user-defined: always matched
        Assert.Equal([1], t.Encode("<s>", addSpecial: false, parseSpecial: true));
        Assert.True(t.IsEndOfGeneration(2));
        Assert.True(t.IsEndOfGeneration(12));
    }

    [Fact]
    public void DecodeDropsOnlyTheDummyPrefix()
    {
        var t = Make();
        Assert.Equal("ab abc", t.Decode(t.Encode("ab abc", addSpecial: false)));
        Assert.Equal("ab", t.Decode([10]));          // first piece loses its leading space
        Assert.Equal("<s> ab", t.Decode([1, 10]));   // after a rendered BOS the space stays
    }
}
