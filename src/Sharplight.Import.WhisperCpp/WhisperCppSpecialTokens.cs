using System.Globalization;

namespace Sharplight.Import.WhisperCpp;

/// <summary>The special token ids of a whisper.cpp vocabulary, as whisper.cpp derives them from the vocabulary size.</summary>
internal sealed class WhisperCppSpecialTokens
{
    /// <summary>Language codes in Whisper's token order (Cantonese last, only in 100-language vocabularies).</summary>
    public static IReadOnlyList<string> Languages { get; } =
    [
        "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el",
        "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az",
        "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af",
        "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as",
        "tt", "haw", "ln", "ha", "ba", "jw", "su", "yue",
    ];

    private readonly WhisperCppHyperParameters _hp;

    /// <summary>Derives the ids from the vocabulary size and language count.</summary>
    public WhisperCppSpecialTokens(WhisperCppHyperParameters hp)
    {
        _hp = hp;
        int shift = hp.IsMultilingual ? 1 : 0;
        int languageShift = hp.IsMultilingual ? hp.LanguageCount - 98 : 0;
        EndOfText = 50256 + shift;
        StartOfTranscript = 50257 + shift;
        Translate = 50357 + languageShift;
        Transcribe = 50358 + languageShift;
        StartOfLanguageModel = 50359 + languageShift;
        StartOfPrevious = 50360 + languageShift;
        NoSpeech = 50361 + languageShift;
        NoTimestamps = 50362 + languageShift;
        TimestampBegin = 50363 + languageShift;
    }

    /// <summary>End of text (also the end of generation).</summary>
    public int EndOfText { get; }

    /// <summary>Start of transcript, the first prompt token.</summary>
    public int StartOfTranscript { get; }

    /// <summary>Task token: translate into English.</summary>
    public int Translate { get; }

    /// <summary>Task token: transcribe.</summary>
    public int Transcribe { get; }

    /// <summary>Start of a language-model prompt (unused by recognition).</summary>
    public int StartOfLanguageModel { get; }

    /// <summary>Introduces previous text before the start token.</summary>
    public int StartOfPrevious { get; }

    /// <summary>No-speech token (its probability after the start token).</summary>
    public int NoSpeech { get; }

    /// <summary>Asks for text without timestamps.</summary>
    public int NoTimestamps { get; }

    /// <summary>The first timestamp token (0.00 s).</summary>
    public int TimestampBegin { get; }

    /// <summary>The Whisper name of a non-text token.</summary>
    public string Name(int id)
    {
        if (id >= TimestampBegin)
        {
            return string.Create(CultureInfo.InvariantCulture, $"<|{(id - TimestampBegin) * 0.02:F2}|>");
        }

        if (id > StartOfTranscript && id < Translate)
        {
            int language = id - StartOfTranscript - 1;
            return language < Languages.Count ? $"<|{Languages[language]}|>" : $"<|extra_{id}|>";
        }

        return id switch
        {
            _ when id == EndOfText => "<|endoftext|>",
            _ when id == StartOfTranscript => "<|startoftranscript|>",
            _ when id == Translate => "<|translate|>",
            _ when id == Transcribe => "<|transcribe|>",
            _ when id == StartOfLanguageModel => "<|startoflm|>",
            _ when id == StartOfPrevious => "<|startofprev|>",
            _ when id == NoSpeech => _hp.IsMultilingual ? "<|nospeech|>" : "<|nocaptions|>",
            _ when id == NoTimestamps => "<|notimestamps|>",
            _ => $"<|extra_{id}|>",
        };
    }
}
