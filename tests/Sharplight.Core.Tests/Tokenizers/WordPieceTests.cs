using Sharplight.Tokenizers.WordPiece;
using Xunit;

namespace Sharplight.Tokenizers.Tests;

/// <summary>Synthetic-vocabulary tests of the "wordpiece" tokenizer; real vocabularies are checked against Hugging Face in
/// Sharplight.Embeddings.Tests.</summary>
public class WordPieceTests
{
    private static WordPieceTokenizer Make(bool lowercase = true) => new(new TokenizerData
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
