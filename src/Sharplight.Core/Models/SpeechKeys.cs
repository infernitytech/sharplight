namespace Sharplight.Models;

/// <summary>
/// Canonical metadata of speech-to-text models whose decoder is prompted with control tokens (the Whisper protocol): a start
/// token, an optional language token, a task token, an optional no-timestamps token, then text and timestamp tokens until the end
/// token. Importers translate their format's vocabulary into these keys; the host decoding loop reads only them.
/// </summary>
/// <remarks>
/// Token ids are ints; lists are int or string arrays. Timestamp tokens are the consecutive ids from <see cref="TimestampBeginId"/>
/// upwards, token <c>TimestampBeginId + i</c> marking <c>i · TimestampStep</c> seconds from the start of the window.
/// </remarks>
public static class SpeechKeys
{
    /// <summary>Start of transcript (string <c>&lt;|startoftranscript|&gt;</c>), the first prompt token.</summary>
    public const string StartId = "speech.start_token_id";
    /// <summary>End of text: generation stops here.</summary>
    public const string EndId = "speech.end_token_id";
    /// <summary>Task token: transcribe in the spoken language.</summary>
    public const string TranscribeId = "speech.transcribe_token_id";
    /// <summary>Task token: translate into English (absent for English-only models).</summary>
    public const string TranslateId = "speech.translate_token_id";
    /// <summary>Token that asks for text without timestamp tokens.</summary>
    public const string NoTimestampsId = "speech.no_timestamps_token_id";
    /// <summary>Token whose probability after the start token is the probability that the window holds no speech.</summary>
    public const string NoSpeechId = "speech.no_speech_token_id";
    /// <summary>Token that introduces previous text (a prompt) before the start token.</summary>
    public const string PreviousId = "speech.previous_token_id";
    /// <summary>First timestamp token (0.00 s).</summary>
    public const string TimestampBeginId = "speech.timestamp_begin_token_id";
    /// <summary>Seconds per timestamp token step (float; Whisper: 0.02).</summary>
    public const string TimestampStep = "speech.timestamp_step";
    /// <summary>Language codes (string[]: "en", "de", ...), in the order of <see cref="LanguageIds"/>; absent for English-only models.</summary>
    public const string Languages = "speech.languages";
    /// <summary>Token ids of the languages (int[]), parallel to <see cref="Languages"/>.</summary>
    public const string LanguageIds = "speech.language_token_ids";
    /// <summary>Tokens never generated (int[]; Whisper's non-speech symbols).</summary>
    public const string SuppressIds = "speech.suppress_token_ids";
    /// <summary>Tokens not generated as the first text token (int[]; Whisper: the space and the end token).</summary>
    public const string BeginSuppressIds = "speech.begin_suppress_token_ids";
    /// <summary>Longest text a window may hold, in tokens (int; Whisper: half the decoder context, 224).</summary>
    public const string MaxTextTokens = "speech.max_text_tokens";
}
