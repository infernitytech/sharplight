using Sharplight.Models;
using Sharplight.Tokenizers;

namespace Sharplight.Speech;

/// <summary>
/// The control tokens of a speech decoder prompted with a start token, a language, a task and a timestamp mode (the canonical
/// <see cref="SpeechKeys"/> metadata, which importers translate from their format). Immutable.
/// </summary>
internal sealed class SpeechTokenProtocol
{
    private readonly Dictionary<string, int> _languageIds;
    private readonly Dictionary<int, string> _languageCodes;

    private SpeechTokenProtocol(IReadOnlyDictionary<string, object> metadata)
    {
        StartId = Required(metadata, SpeechKeys.StartId);
        EndId = Required(metadata, SpeechKeys.EndId);
        TimestampBeginId = Required(metadata, SpeechKeys.TimestampBeginId);
        NoTimestampsId = Optional(metadata, SpeechKeys.NoTimestampsId);
        TranscribeId = Optional(metadata, SpeechKeys.TranscribeId);
        TranslateId = Optional(metadata, SpeechKeys.TranslateId);
        NoSpeechId = Optional(metadata, SpeechKeys.NoSpeechId);
        PreviousId = Optional(metadata, SpeechKeys.PreviousId);
        TimestampStep = metadata.TryGetValue(SpeechKeys.TimestampStep, out var step) ? ValueConverter.Convert<float>(step, "Metadata", SpeechKeys.TimestampStep) : 0.02f;
        MaxTextTokens = Optional(metadata, SpeechKeys.MaxTextTokens) ?? 224;
        MaxInitialTimestamp = Optional(metadata, SpeechKeys.MaxInitialTimestampIndex);
        SuppressIds = TokenizerMetadataReader.GetIntArray(metadata, SpeechKeys.SuppressIds) ?? [];
        BeginSuppressIds = TokenizerMetadataReader.GetIntArray(metadata, SpeechKeys.BeginSuppressIds) ?? [];
        string[] languages = TokenizerMetadataReader.GetStringArray(metadata, SpeechKeys.Languages) ?? [];
        int[] languageIds = TokenizerMetadataReader.GetIntArray(metadata, SpeechKeys.LanguageIds) ?? [];
        if (languages.Length != languageIds.Length)
        {
            throw new InvalidDataException($"{SpeechKeys.Languages} has {languages.Length} entries but {SpeechKeys.LanguageIds} has {languageIds.Length}.");
        }

        _languageIds = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        _languageCodes = [];
        for (int i = 0; i < languages.Length; i++)
        {
            _languageIds[languages[i]] = languageIds[i];
            _languageCodes[languageIds[i]] = languages[i];
        }

        LanguageIds = languageIds;
    }

    /// <summary>First prompt token.</summary>
    public int StartId { get; }

    /// <summary>End of the text: decoding stops.</summary>
    public int EndId { get; }

    /// <summary>The first timestamp token (0 s from the window start); every id from here up is a timestamp.</summary>
    public int TimestampBeginId { get; }

    /// <summary>Token asking for text without timestamps, if the model has one.</summary>
    public int? NoTimestampsId { get; }

    /// <summary>Task token for transcription, if the model has task tokens.</summary>
    public int? TranscribeId { get; }

    /// <summary>Task token for translation into English, if the model can translate.</summary>
    public int? TranslateId { get; }

    /// <summary>Token whose probability after the start token is the no-speech probability, if the model has one.</summary>
    public int? NoSpeechId { get; }

    /// <summary>Token that introduces previous text before the start token, if the model has one.</summary>
    public int? PreviousId { get; }

    /// <summary>Seconds per timestamp step.</summary>
    public float TimestampStep { get; }

    /// <summary>Most text tokens per window.</summary>
    public int MaxTextTokens { get; }

    /// <summary>Latest timestamp step the first token of a window may have, or null for no limit.</summary>
    public int? MaxInitialTimestamp { get; }

    /// <summary>Tokens never generated.</summary>
    public int[] SuppressIds { get; }

    /// <summary>Tokens not generated as the first token.</summary>
    public int[] BeginSuppressIds { get; }

    /// <summary>Token ids of the languages; empty for single-language models.</summary>
    public int[] LanguageIds { get; }

    /// <summary>True when the prompt carries a language token.</summary>
    public bool IsMultilingual => LanguageIds.Length > 0;

    /// <summary>True when <paramref name="metadata"/> describes a prompted speech decoder.</summary>
    /// <param name="metadata">Canonical metadata of the decoder.</param>
    public static bool IsPresent(IReadOnlyDictionary<string, object> metadata)
        => metadata.ContainsKey(SpeechKeys.StartId) && metadata.ContainsKey(SpeechKeys.EndId) && metadata.ContainsKey(SpeechKeys.TimestampBeginId);

    /// <summary>Reads the protocol from canonical metadata.</summary>
    /// <param name="metadata">Canonical metadata of the decoder.</param>
    /// <exception cref="InvalidDataException">A required key is missing.</exception>
    public static SpeechTokenProtocol From(IReadOnlyDictionary<string, object> metadata) => new(metadata);

    /// <summary>The token of a language code (ISO 639-1, case-insensitive).</summary>
    /// <param name="code">The language code.</param>
    /// <exception cref="ArgumentException">The model does not know the language.</exception>
    public int LanguageToken(string code)
        => _languageIds.TryGetValue(code, out int id)
            ? id
            : throw new ArgumentException($"The model does not support the language '{code}' (supported: {string.Join(", ", _languageIds.Keys)}).");

    /// <summary>The code of a language token, or null.</summary>
    /// <param name="token">A token id.</param>
    public string? LanguageCode(int token) => _languageCodes.TryGetValue(token, out var code) ? code : null;

    /// <summary>The prompt that starts a window: start, language, task and (without timestamps) the no-timestamps token.</summary>
    /// <param name="language">The language token, or null for single-language models.</param>
    /// <param name="translate">Translate instead of transcribing.</param>
    /// <param name="timestamps">Generate timestamp tokens.</param>
    /// <exception cref="NotSupportedException">Translation was asked of a model without a translate token.</exception>
    public List<int> StartSequence(int? language, bool translate, bool timestamps)
    {
        List<int> prompt = [StartId];
        if (language is int languageToken)
        {
            prompt.Add(languageToken);
        }

        if (translate)
        {
            prompt.Add(TranslateId ?? throw new NotSupportedException("The model cannot translate (it has no translate task token)."));
        }
        else if (TranscribeId is int transcribe)
        {
            prompt.Add(transcribe);
        }

        if (!timestamps && NoTimestampsId is int noTimestamps)
        {
            prompt.Add(noTimestamps);
        }

        return prompt;
    }

    private static int Required(IReadOnlyDictionary<string, object> metadata, string key)
        => Optional(metadata, key) ?? throw new InvalidDataException($"Speech decoder metadata '{key}' is missing.");

    private static int? Optional(IReadOnlyDictionary<string, object> metadata, string key) => TokenizerMetadataReader.GetInt(metadata, key);
}
