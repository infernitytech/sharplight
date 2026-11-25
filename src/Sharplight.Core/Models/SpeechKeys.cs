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
    /// <summary>Latest timestamp step the first token of a window may have (int; Whisper: 50, i.e. 1 s); absent: no limit.</summary>
    public const string MaxInitialTimestampIndex = "speech.max_initial_timestamp_index";
    /// <summary>Longest text a window may hold, in tokens (int; Whisper: half the decoder context, 224).</summary>
    public const string MaxTextTokens = "speech.max_text_tokens";
    /// <summary>
    /// The decoder prompt of models whose prompt is a fixed template (Canary), as token ids (int[]): -1 stands for the source language's
    /// token, -2 for the target language's (the source language, or English when translating). Without it the prompt is start, language,
    /// task and no-timestamps tokens (Whisper).
    /// </summary>
    public const string PromptTemplateIds = "speech.prompt_template_ids";
    /// <summary>The language (ISO 639-1) a multilingual model is prompted with when the request names none and the model cannot detect it (string).</summary>
    public const string DefaultLanguage = "speech.default_language";
    /// <summary>
    /// Longest chunk of a recording, in seconds (float), for models whose reference splits long recordings on the waveform (Cohere Transcribe:
    /// 35): a longer recording is cut into chunks of at most this length, each with its own features (and feature normalization) and decoded
    /// on its own. Absent: the features of the whole recording are cut into encoder windows.
    /// </summary>
    public const string ChunkSeconds = "speech.chunk_seconds";
    /// <summary>
    /// How far before a chunk's end the cut is searched, in seconds (float, with <see cref="ChunkSeconds"/>): the chunk ends at the start of the
    /// quietest <see cref="ChunkEnergyWindow"/> (lowest RMS) in this stretch. Absent or 0: chunks end at <see cref="ChunkSeconds"/>.
    /// </summary>
    public const string ChunkSearchSeconds = "speech.chunk_search_seconds";
    /// <summary>The samples of the RMS windows the chunk cut is searched with (int, with <see cref="ChunkSearchSeconds"/>; default 1600, 0.1 s at 16 kHz).</summary>
    public const string ChunkEnergyWindow = "speech.chunk_energy_window";
    /// <summary>Languages (ISO 639-1, string[]) whose chunk texts join without a space (Cohere Transcribe: "ja", "zh"); other languages join with one.</summary>
    public const string UnspacedLanguages = "speech.unspaced_languages";
}
