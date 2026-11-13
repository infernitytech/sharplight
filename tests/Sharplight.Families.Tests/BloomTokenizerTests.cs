using Sharplight.Gguf;
using Sharplight.Tokenizers;

namespace Sharplight.Families.Tests;

/// <summary>
/// The BLOOM pre-tokenizer preset against llama.cpp (b11177, CPU): the token ids its <c>/tokenize</c> endpoint returns for the test strings of
/// llama.cpp's own tokenizer tests (<c>Fixtures/bloom-vocab.inp</c>, the same strings as the Falcon fixture), with the vocabulary of
/// <c>~/models/gguf/bloom-560m/bloom-560m.Q4_K_M.gguf</c>; the test skips without that file.
/// </summary>
public class BloomTokenizerTests
{
    private const string Separator = "\n__ggml_vocab_test__\n";

    private static readonly string VocabPath = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "models", "gguf", "bloom-560m", "bloom-560m.Q4_K_M.gguf");

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

        string input = File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Fixtures", "bloom-vocab.inp"));
        string[] cases = input.Split(Separator);
        string[] expected = File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Fixtures", "bloom-vocab.out")).Split('\n');
        Assert.Equal("bloom", (string)file.Metadata["tokenizer.ggml.pre"]);
        for (int i = 0; i < expected.Length - 1; i++)
        {
            int[] ids = tokenizer.Encode(cases[i], addSpecial: false, parseSpecial: false);
            int[] want = expected[i].Split(' ', StringSplitOptions.RemoveEmptyEntries).Select(int.Parse).ToArray();
            Assert.True(want.SequenceEqual(ids), $"case {i}: {cases[i].ReplaceLineEndings("\\n")} expected [{string.Join(' ', want)}] got [{string.Join(' ', ids)}]");
        }
    }
}
