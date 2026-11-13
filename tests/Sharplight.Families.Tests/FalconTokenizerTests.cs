using Sharplight.Gguf;
using Sharplight.Tokenizers;

namespace Sharplight.Families.Tests;

/// <summary>
/// The Falcon pre-tokenizer preset against llama.cpp's own tokenizer test (<c>models/ggml-vocab-falcon.gguf</c> with its <c>.inp</c>/<c>.out</c>
/// files, kept in Fixtures). The vocabulary file is expected at <c>~/models/gguf/ggml-vocab/ggml-vocab-falcon.gguf</c>; the test skips without it.
/// </summary>
public class FalconTokenizerTests
{
    private const string Separator = "\n__ggml_vocab_test__\n";

    private static readonly string VocabPath = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models", "gguf", "ggml-vocab", "ggml-vocab-falcon.gguf");

    [Fact]
    public void Encoding_matches_llama_cpp()
    {
        Assert.SkipUnless(File.Exists(VocabPath), $"{VocabPath} not present");
        using var file = GgufFile.Open(VocabPath);
        var tokenizer = TokenizerFactory.Create(new TokenizerData
        {
            Model = "gpt2",
            PreTokenizer = (string)file.Metadata["tokenizer.ggml.pre"],
            Tokens = (string[])file.Metadata["tokenizer.ggml.tokens"],
            Merges = (string[])file.Metadata["tokenizer.ggml.merges"],
        });

        string input = File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Fixtures", "ggml-vocab-falcon.inp"));
        string[] cases = input.Split(Separator);
        string[] expected = File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Fixtures", "ggml-vocab-falcon.out")).Split('\n');
        Assert.Equal("falcon", (string)file.Metadata["tokenizer.ggml.pre"]);
        for (int i = 0; i < expected.Length - 1; i++)
        {
            int[] ids = tokenizer.Encode(cases[i], addSpecial: false, parseSpecial: false);
            int[] want = expected[i].Split(' ', StringSplitOptions.RemoveEmptyEntries).Select(int.Parse).ToArray();
            Assert.True(want.SequenceEqual(ids), $"case {i}: {cases[i].ReplaceLineEndings("\\n")} expected [{string.Join(' ', want)}] got [{string.Join(' ', ids)}]");
        }
    }
}
