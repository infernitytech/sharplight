using Sharplight.Import.WhisperCpp;
using Sharplight.Models;
using Sharplight.Tokenizers;
using Xunit;

namespace Sharplight.Speech.Tests;

/// <summary>
/// The tokenizer rebuilt from a whisper.cpp vocabulary (ranks only, merges recovered) against the Hugging Face tokenizer of the same
/// model: identical token strings, identical encodings, and the same suppressed and special token ids.
/// </summary>
public sealed class WhisperCppVocabularyTests
{
    private const string Sample = "And so, my fellow Americans: ask not what your country can do for you — ask what you can do for your country! 1961 «Über» 東京";

    /// <summary>Tokens, merges-based encodings and speech keys agree with the transformers checkpoint.</summary>
    [Theory]
    [InlineData("ggml-tiny.en", "whisper-tiny.en")]
    [InlineData("ggml-tiny", "whisper-tiny")]
    public void VocabularyMatchesHuggingFace(string whisperCpp, string huggingFace)
    {
        string binPath = SpeechEnvironment.WhisperCpp(whisperCpp);
        string hfPath = SpeechEnvironment.HuggingFace(huggingFace);
        if (!File.Exists(binPath) || !Directory.Exists(hfPath))
        {
            Assert.Skip($"{binPath} or {hfPath} is missing.");
        }

        var bin = WhisperCppImporter.Open(binPath);
        var hf = Import.HuggingFace.HuggingFaceImporter.Open(hfPath);
        try
        {
            var binTokenizer = TokenizerFactory.Create(TokenizerData.FromMetadata(bin.Metadata));
            var hfTokenizer = TokenizerFactory.Create(TokenizerData.FromMetadata(hf.Metadata));
            Assert.Equal(hfTokenizer.Encode(Sample, addSpecial: false, parseSpecial: false), binTokenizer.Encode(Sample, addSpecial: false, parseSpecial: false));
            foreach (string key in new[] { SpeechKeys.StartId, SpeechKeys.EndId, SpeechKeys.NoTimestampsId, SpeechKeys.TimestampBeginId, SpeechKeys.PreviousId,
                SpeechKeys.NoSpeechId, SpeechKeys.TranscribeId, SpeechKeys.TranslateId })
            {
                Assert.Equal(hf.Metadata.GetValueOrDefault(key), bin.Metadata.GetValueOrDefault(key));
            }

            Assert.Equal((int[])hf.Metadata[SpeechKeys.SuppressIds], (int[])bin.Metadata[SpeechKeys.SuppressIds]);
            Assert.Equal((int[])hf.Metadata[SpeechKeys.BeginSuppressIds], (int[])bin.Metadata[SpeechKeys.BeginSuppressIds]);
            if (hf.Metadata.TryGetValue(SpeechKeys.Languages, out var languages))
            {
                Assert.Equal((string[])languages, (string[])bin.Metadata[SpeechKeys.Languages]);
            }

            int timestamp = (int)bin.Metadata[SpeechKeys.TimestampBeginId] + 150;
            Assert.Equal(hfTokenizer.TokenString(timestamp), binTokenizer.TokenString(timestamp));
        }
        finally
        {
            foreach (var resource in bin.OwnedResources.Concat(hf.OwnedResources))
            {
                resource.Dispose();
            }
        }
    }
}
