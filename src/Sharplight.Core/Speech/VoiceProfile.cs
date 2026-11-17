namespace Sharplight.Speech;

/// <summary>
/// A reusable voice of a voice-cloning speech model: the conditioning the model derived from a recording (speaker embeddings, prompt tokens, prompt
/// mel, ...), so later requests skip the recording's encoders and the recording itself. Created by <see cref="IVoiceCloner"/>, used through
/// <see cref="SpeechSynthesisOptions.VoiceProfile"/> with a model of the same <see cref="Architecture"/>, stored by <see cref="VoiceLibrary"/>.
/// </summary>
public sealed class VoiceProfile
{
    /// <summary>The voice's name.</summary>
    public required string Name { get; init; }

    /// <summary>The architecture family the voice belongs to (<c>xtts</c>, <c>chatterbox</c>, <c>cosyvoice3</c>, <c>f5-tts</c>): conditioning is not portable between models.</summary>
    public required string Architecture { get; init; }

    /// <summary>The words spoken in the recording, for models that align them with the recording (CosyVoice 3 zero-shot, F5-TTS), or <see langword="null"/>.</summary>
    public string? ReferenceText { get; init; }

    /// <summary>When the voice was created.</summary>
    public DateTimeOffset Created { get; init; } = DateTimeOffset.UtcNow;

    /// <summary>The sample rate of the model that derived the voice (for information).</summary>
    public int SampleRate { get; init; }

    /// <summary>The conditioning arrays by name.</summary>
    public required IReadOnlyDictionary<string, VoiceTensor> Tensors { get; init; }

    /// <summary>A copy of the voice under another name.</summary>
    /// <param name="name">The new name.</param>
    public VoiceProfile Renamed(string name) => new() { Name = name, Architecture = Architecture, ReferenceText = ReferenceText, Created = Created, SampleRate = SampleRate, Tensors = Tensors };
}
