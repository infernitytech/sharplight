using Sharplight.Tokenizers.Unigram;
using Xunit;

namespace Sharplight.Tokenizers.Tests;

/// <summary>Synthetic-vocabulary tests of the "unigram" tokenizer.</summary>
public class UnigramTests
{
    // ids: 0 <s>, 1 <pad>, 2 </s>, 3 <unk>, then pieces with log-probabilities.
    private static UnigramTokenizer Make() => new(new TokenizerData
    {
        Model = "unigram",
        Tokens = ["<s>", "<pad>", "</s>", "<unk>", "▁", "▁hello", "▁he", "llo", "l", "o", "h", "e", "▁world", "▁wor", "ld", "!"],
        Scores = [0, 0, 0, 0, -3, -5, -2, -2, -4, -4, -4, -4, -3, -2, -2, -3],
        TokenTypes = [3, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        BosId = 0,
        EosId = 2,
        PadId = 1,
        UnkId = 3,
    });

    [Fact]
    public void ViterbiPicksMostProbableSegmentation()
    {
        var t = Make();
        // ▁he + llo (-4) beats ▁hello (-5); ▁world (-3) beats ▁wor + ld (-4).
        Assert.Equal([0, 6, 7, 12, 2], t.Encode("hello world"));
        Assert.Equal([6, 7, 12, 15], t.Encode("hello  world!", addSpecial: false));
    }

    [Fact]
    public void UncoveredCharactersFuseIntoOneUnknown()
        => Assert.Equal([6, 7, 4, 3], Make().Encode("hello ×÷", addSpecial: false));

    [Fact]
    public void DecodeRestoresSpaces()
    {
        var t = Make();
        Assert.Equal("hello world!", t.Decode(t.Encode("hello world!"), skipSpecial: true));
    }
}
