using Sharplight.Tokenizers;
using Xunit;

namespace Sharplight.Import.HuggingFace.Tests;

/// <summary>
/// Kimi's tiktoken tokenizer (<c>tiktoken.model</c> and <c>tokenization_kimi.py</c>, no tokenizer.json) through the Hugging Face importer: the
/// ids of the tiny-random/kimi-k3 checkpoint's own remote tokenizer (transformers 5.20, tiktoken 0.14) for English with contractions and
/// case changes, Chinese and Japanese runs (Han split from kana and Latin letters), digit groups, whitespace, code, accented and non-Latin
/// letters, emoji and supplementary-plane Han. Skipped when the checkpoint is missing.
/// </summary>
public sealed class KimiTiktokenTests
{
    private static readonly string Directory = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models", "hf", "tiny-random-kimi-k3");

    public static TheoryData<string, int[]> Cases() => new()
    {
        { "The capital of France is a city that, in the year 1900 and beyond, was known for many things such as art",
            [1008, 10484, 318, 15383, 387, 261, 5243, 473, 11, 306, 276, 1389, 220, 8684, 15, 316, 10579, 11, 743, 5624, 395, 2133, 3456, 2266, 559, 2718] },
        { "Hello WORLD, it's McDonald's! They'll say: \"don't\" — I'VE seen 1234567 and 3.14159.",
            [19180, 101782, 11, 4643, 8315, 43610, 1048, 0, 3350, 6683, 3637, 25, 414, 88709, 1, 4275, 4154, 4571, 6219, 220, 6694, 12972, 22, 316, 220, 18, 13, 18702, 6276, 13] },
        { "中文分词测试：我爱北京天安门。混合English和中文123个", [16717, 671, 3961, 9555, 2648, 33085, 3372, 91676, 292, 13935, 44372, 488, 16717, 6694, 439] },
        { "日本語のテキスト、カタカナとひらがな。漢字も",
            [4546, 63011, 4663, 25875, 32045, 33360, 343, 26121, 21449, 26121, 42125, 9219, 127935, 16800, 10389, 11839, 292, 1435, 95, 1855, 20624] },
        { "  leading spaces\n\n\ttabs\r\nand trailing   ", [220, 8134, 14803, 382, 5604, 5609, 462, 516, 42711, 274] },
        { "def f(x):\n    return x ** 2  # comment\n", [1166, 285, 3940, 2360, 274, 745, 1288, 3465, 220, 17, 220, 1214, 8018, 198] },
        { "naïve café Ünïcödé ΑΒΓ αβγ Привет мир",
            [4768, 44187, 367, 70591, 61412, 77, 44187, 66, 5954, 67, 1941, 82394, 138, 240, 115528, 6235, 24866, 20942, 113419, 63694, 5993, 17898] },
        { "emoji 🙂🚀 and 𠀀𠀁 rare Han", [33120, 56774, 5885, 40101, 316, 220, 172, 254, 222, 222, 172, 254, 222, 223, 13234, 32229] },
        { "x=[1,2,3];y={'a':1}", [87, 16361, 16, 11, 17, 11, 18, 11340, 88, 40810, 64, 2764, 16, 92] },
    };

    private static ITokenizer Tokenizer()
    {
        Assert.SkipUnless(System.IO.Directory.Exists(Directory), $"{Directory} not present");
        var source = HuggingFaceImporter.Open(Directory);
        try
        {
            return TokenizerFactory.Create(TokenizerData.FromMetadata(source.Metadata));
        }
        finally
        {
            foreach (var resource in source.OwnedResources)
            {
                resource.Dispose();
            }
        }
    }

    [Theory, MemberData(nameof(Cases))]
    public void Encode_matches_the_remote_tokenizer(string text, int[] ids)
    {
        var tokenizer = Tokenizer();
        Assert.Equal(ids, tokenizer.Encode(text, addSpecial: false, parseSpecial: false));
        Assert.Equal(text, tokenizer.Decode(ids));
    }

    [Fact]
    public void Special_tokens_follow_tokenizer_config()
    {
        var tokenizer = Tokenizer();
        Assert.Equal(163840, tokenizer.VocabSize);
        Assert.Equal(163584, tokenizer.BosId);
        Assert.Equal(163585, tokenizer.EosId);
        Assert.Equal(163586, tokenizer.TokenId("<|end_of_msg|>"));
        Assert.Equal(163600, tokenizer.TokenId("<|reserved_token_163600|>"));
        Assert.Equal([18699], tokenizer.Encode("Hi"));
    }
}
