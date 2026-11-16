using Sharplight.Tokenizers;
using Xunit;

namespace Sharplight.Import.HuggingFace.Tests;

/// <summary>
/// A "slow" byte-level BPE tokenizer saved without tokenizer.json (vocab.json + merges.txt + tokenizer_config.json, as facebook/opt-125m
/// ships it) through the Hugging Face importer: the ids transformers 5.23 gives (it converts the files with its GPT2Converter), and
/// the BOS that OPT's <c>add_bos_token</c> prepends. Uses tiny-random-opt (<c>tools/mk_tiny_opt.py</c>, opt-125m's tokenizer files);
/// skipped when it is missing.
/// </summary>
public sealed class SlowBpeTokenizerTests
{
    private static readonly string Directory = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models", "hf", "tiny-random-opt");

    public static TheoryData<string, int[]> Cases() => new()
    {
        { "Hello world! The quick brown fox jumps over the lazy dog. 1234567890 3.14159 2026-11-16",
            [31414, 232, 328, 20, 2119, 6219, 23602, 13855, 81, 5, 22414, 2335, 4, 17072, 1898, 37628, 3248, 155, 4, 1570, 27129, 291, 2481, 12, 1225, 12, 1549] },
        { "Code: def f(x):\n    return x**2  # comment\n\n\tTabs\tand  multiple   spaces   \n\n\n",
            [41555, 35, 3816, 856, 1640, 1178, 3256, 50118, 1437, 1437, 1437, 671, 3023, 12606, 176, 1437, 849, 1129, 50140, 50117, 565, 10155, 50117, 463, 1437, 1533, 1437, 1437, 5938, 1437, 1437, 1437, 50140, 50118] },
        { "Emoji 🤖🚀👍🏽 and symbols ©®™ ∑∫√ €£¥ <tag attr=\"v\">&amp;</tag>",
            [717, 4992, 5186, 8103, 10470, 25448, 6569, 15113, 7471, 31193, 8384, 6569, 9357, 10809, 8, 19830, 3705, 8716, 13445, 47444, 3602, 48104, 4958, 48104, 15113, 4480, 29254, 4056, 8210, 28696, 10058, 15095, 338, 40635, 705, 46479, 947, 3914, 131, 49138, 10058, 15698] },
        { "don't won't I'm they're CAPS lowercase MiXeD 00042 1,000,000", [7254, 75, 351, 75, 38, 437, 51, 214, 7778, 104, 795, 11173, 9109, 1000, 242, 495, 12096, 3714, 112, 6, 151, 6, 151] },
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
    public void Encode_matches_transformers(string text, int[] ids)
    {
        var tokenizer = Tokenizer();
        Assert.Equal(ids, tokenizer.Encode(text, addSpecial: false, parseSpecial: false));
        Assert.Equal(text, tokenizer.Decode(ids));
    }

    [Fact]
    public void Bos_is_added_as_configured()
    {
        var tokenizer = Tokenizer();
        Assert.Equal(2, tokenizer.BosId);
        Assert.Equal(2, tokenizer.EosId);
        Assert.Equal([2, 133, 812, 9, 1470, 16], tokenizer.Encode("The capital of France is"));
    }
}
