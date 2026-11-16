using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// How an audio language model is prompted for a transcription (the importer writes it, the recognizer reads it from the decoder component, so
/// saved packages keep it): a template with the placeholders <c>{context}</c> (the request's prompt text, may be empty), <c>{audio}</c>
/// (replaced by one audio placeholder token per audio embedding row) and optionally <c>{language}</c> (where the language goes; the end when
/// absent), the text there when a language is given (<c>{language}</c> in it is the language's English name, <c>{code}</c> its ISO code, see
/// <see cref="Speech.SpeechLanguages"/>) and when none is given, and the marker after which the generated text is the transcript
/// (what comes before it names the detected language).
/// </summary>
/// <remarks>
/// Keys under <c>{arch}.audio.prompt.</c>: <c>template</c>, <c>language_prefix</c>, <c>auto_language</c>, <c>transcript_marker</c> and
/// <c>{arch}.audio.token_id</c> (the placeholder token).
/// </remarks>
internal sealed record AudioPromptSettings
{
    /// <summary>The template with <c>{context}</c> and <c>{audio}</c>.</summary>
    public required string Template { get; init; }

    /// <summary>The assistant prefix that forces a language (<c>{language}</c>), or empty when the model has none.</summary>
    public string LanguagePrefix { get; init; } = "";

    /// <summary>The text at the template's <c>{language}</c> placeholder when the request gives no language (empty by default).</summary>
    public string AutoLanguage { get; init; } = "";

    /// <summary>The marker that ends the language preamble of the output, or empty when the output is the transcript only.</summary>
    public string TranscriptMarker { get; init; } = "";

    /// <summary>The audio placeholder token.</summary>
    public required int AudioToken { get; init; }

    /// <summary>The settings of a source, or <see langword="null"/> when it has no audio prompt.</summary>
    /// <param name="src">The model source.</param>
    public static AudioPromptSettings? From(ModelSource src)
    {
        string template = src.Arch("audio.prompt.template", "");
        if (template.Length == 0)
        {
            return null;
        }

        return new AudioPromptSettings
        {
            Template = template,
            LanguagePrefix = src.Arch("audio.prompt.language_prefix", ""),
            AutoLanguage = src.Arch("audio.prompt.auto_language", ""),
            TranscriptMarker = src.Arch("audio.prompt.transcript_marker", ""),
            AudioToken = src.Arch<int>("audio.token_id"),
        };
    }

    /// <summary>The settings stored in a component's configuration.</summary>
    /// <param name="component">The decoder component.</param>
    public static AudioPromptSettings FromComponent(ModelComponent component) => new()
    {
        Template = component.ConfigValue("promptTemplate", ""),
        LanguagePrefix = component.ConfigValue("languagePrefix", ""),
        AutoLanguage = component.ConfigValue("autoLanguage", ""),
        TranscriptMarker = component.ConfigValue("transcriptMarker", ""),
        AudioToken = component.ConfigValue("audioToken", 0),
    };

    /// <summary>Whether a decoder component carries these settings.</summary>
    /// <param name="component">The decoder component.</param>
    public static bool IsPresent(ModelComponent component) => component.ConfigValue("promptTemplate", "").Length > 0;

    /// <summary>The settings as component configuration values.</summary>
    public IReadOnlyDictionary<string, object> ComponentConfig() => new Dictionary<string, object>(StringComparer.Ordinal)
    {
        ["promptTemplate"] = Template,
        ["languagePrefix"] = LanguagePrefix,
        ["autoLanguage"] = AutoLanguage,
        ["transcriptMarker"] = TranscriptMarker,
        ["audioToken"] = AudioToken,
    };
}
