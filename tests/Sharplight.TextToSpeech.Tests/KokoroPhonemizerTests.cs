using Sharplight.Architectures;

namespace Sharplight.TextToSpeech.Tests;

/// <summary>The English front end of Kokoro against the phonemes of misaki (the front end the model was trained with) for sentences without part-of-speech dependent choices.</summary>
public class KokoroPhonemizerTests
{
    private static KokoroPhonemizer Create(string variant)
    {
        var lexicons = Path.Combine(TtsModels.Kokoro, "lexicon");
        TtsModels.Require(Path.Combine(lexicons, variant + "_gold.json"), Path.Combine(lexicons, variant + "_silver.json"));
        var gold = new KokoroLexicon();
        gold.Add(File.ReadAllBytes(Path.Combine(lexicons, variant + "_gold.json")));
        var silver = new KokoroLexicon();
        silver.Add(File.ReadAllBytes(Path.Combine(lexicons, variant + "_silver.json")));
        return new KokoroPhonemizer(gold, silver);
    }

    [Theory]
    [InlineData("Hello world.", "həlˈO wˈɜɹld.")]
    [InlineData("The quick brown fox jumps over the lazy dog.", "ðə kwˈɪk bɹˈWn fˈɑks ʤˈʌmps ˈOvəɹ ðə lˈAzi dˈɔɡ.")]
    [InlineData("What time is it, my friend?", "wˌʌt tˈIm ɪz ɪt, mI fɹˈɛnd?")]
    public void American_phonemes_follow_the_lexicon(string text, string expected)
    {
        var phonemizer = Create("us");
        Assert.Equal(expected, string.Concat(phonemizer.Chunks(text, null, 400)));
    }

    [Fact]
    public void Chunks_split_long_passages_at_word_boundaries()
    {
        var phonemizer = Create("us");
        var chunks = phonemizer.Chunks(string.Join(' ', Enumerable.Repeat("The quick brown fox jumps over the lazy dog.", 6)), null, 120).ToList();
        Assert.True(chunks.Count > 1);
        Assert.All(chunks, chunk => Assert.InRange(chunk.Length, 1, 120));
    }

    [Fact]
    public void Unknown_words_get_rule_based_phonemes_and_numbers_are_spelled()
    {
        var phonemizer = Create("us");
        string phonemes = phonemizer.Phonemize("Zorbleflax 25");
        Assert.Contains("twˈɛnti", phonemes, StringComparison.Ordinal);
        Assert.StartsWith("zˈ", phonemes, StringComparison.Ordinal);
    }
}
