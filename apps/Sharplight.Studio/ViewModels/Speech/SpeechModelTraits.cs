using Sharplight.Speech;

namespace Sharplight.Studio.ViewModels.Speech;

/// <summary>
/// What the Speak page offers for a text-to-speech model, from what its client says it can do (its canonical architecture, the language list it implements).
/// </summary>
/// <param name="MultiSpeaker">The text is a script of <c>Speaker N: ...</c> turns with a voice per speaker.</param>
/// <param name="ScriptInOneRequest">The model speaks the whole script in one request (VibeVoice); otherwise the page speaks the turns one by one (CSM).</param>
/// <param name="DiffusionOptions">The model has a diffusion head with steps and guidance (VibeVoice).</param>
/// <param name="VoiceDesign">A text describes the voice to speak with (OmniVoice).</param>
/// <param name="Languages">The language names the model takes, or empty when it has no language choice.</param>
internal sealed record SpeechModelTraits(bool MultiSpeaker, bool ScriptInOneRequest, bool DiffusionOptions, bool VoiceDesign, IReadOnlyList<string> Languages)
{
    /// <summary>A model with no special features.</summary>
    public static SpeechModelTraits None { get; } = new(false, false, false, false, []);

    /// <summary>The traits of a client.</summary>
    /// <param name="client">The leased synthesizer.</param>
    public static SpeechModelTraits For(ISpeechSynthesizer client)
    {
        ArgumentNullException.ThrowIfNull(client);
        var languages = (client as ISpeechLanguageList ?? client.GetService(typeof(ISpeechLanguageList)) as ISpeechLanguageList)?.LanguageNames ?? [];
        return client.Metadata.Architecture switch
        {
            "vibevoice" => new SpeechModelTraits(true, true, true, false, languages),
            "csm" => new SpeechModelTraits(true, false, false, false, languages),
            "omnivoice" => new SpeechModelTraits(false, false, false, true, languages),
            _ => languages.Count > 0 ? None with { Languages = languages } : None,
        };
    }
}
