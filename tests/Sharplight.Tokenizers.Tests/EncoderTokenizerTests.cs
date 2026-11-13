using Sharplight.Tokenizers;
using Sharplight.Tokenizers.Unigram;
using Sharplight.Tokenizers.WordPiece;
using Xunit;

namespace Sharplight.Tokenizers.Tests;

/// <summary>Synthetic-vocabulary tests of the encoder tokenizers ("wordpiece", "unigram"); real vocabularies are checked against
/// Hugging Face in Sharplight.Embeddings.Tests.</summary>
public class WordPieceTests
{
    static WordPieceTokenizer Make(bool lowercase = true) => new(new TokenizerData
    {
        Model = "wordpiece",
        Tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "un", "##aff", "##able", "hello", "world", ",", "!", "cafe", "東", "京", "Hello"],
        Lowercase = lowercase,
    });

    [Fact]
    public void GreedyLongestMatchWithContinuations()
    {
        var t = Make();
        Assert.Equal([2, 5, 6, 7, 3], t.Encode("unaffable"));
        Assert.Equal([5, 6, 7], t.Encode("unaffable", addSpecial: false));
        Assert.Equal("unaffable", t.Decode([5, 6, 7]));
    }

    [Fact]
    public void LowercasesStripsAccentsAndSplitsPunctuationAndCjk()
    {
        var t = Make();
        Assert.Equal([8, 10, 9, 11], t.Encode("Hello,  WORLD!", addSpecial: false));
        Assert.Equal([12], t.Encode("Café", addSpecial: false));
        Assert.Equal([13, 14], t.Encode("東京", addSpecial: false));
    }

    [Fact]
    public void UnknownWordBecomesOneUnk()
    {
        var t = Make();
        Assert.Equal([1, 9], t.Encode("unxyz world", addSpecial: false));
    }

    [Fact]
    public void CasedVocabularyKeepsCase()
        => Assert.Equal([15], Make(lowercase: false).Encode("Hello", addSpecial: false));

    [Fact]
    public void SpecialTokensInTextAreMatched()
        => Assert.Equal([8, 4, 9], Make().Encode("hello [MASK] world", addSpecial: false));

    [Fact]
    public void FactoryCreatesIt()
        => Assert.IsType<WordPieceTokenizer>(TokenizerFactory.Create(new TokenizerData { Model = "wordpiece", Tokens = ["[UNK]", "a"] }));
}

public class UnigramTests
{
    // ids: 0 <s>, 1 <pad>, 2 </s>, 3 <unk>, then pieces with log-probabilities.
    static UnigramTokenizer Make() => new(new TokenizerData
    {
        Model = "unigram",
        Tokens = ["<s>", "<pad>", "</s>", "<unk>", "▁", "▁hello", "▁he", "llo", "l", "o", "h", "e", "▁world", "▁wor", "ld", "!"],
        Scores = [0, 0, 0, 0, -3, -5, -2, -2, -4, -4, -4, -4, -3, -2, -2, -3],
        TokenTypes = [3, 3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        BosId = 0, EosId = 2, PadId = 1, UnkId = 3,
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
