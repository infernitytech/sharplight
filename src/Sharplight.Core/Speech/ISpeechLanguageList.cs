namespace Sharplight.Speech;

/// <summary>
/// Implemented by speech synthesizers that speak a long list of languages and take the language by name (OmniVoice: more than 600), so that a UI can offer the list.
/// Reached through <see cref="Tasks.ITaskClient.GetService"/> or by testing the client.
/// </summary>
public interface ISpeechLanguageList
{
    /// <summary>The language names a request accepts as <see cref="SpeechSynthesisOptions.Language"/> (lower case, alphabetical).</summary>
    public IReadOnlyList<string> LanguageNames { get; }
}
