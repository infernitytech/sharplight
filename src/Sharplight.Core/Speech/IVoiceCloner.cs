using Sharplight.Audio;

namespace Sharplight.Speech;

/// <summary>
/// Implemented by speech synthesizers that clone a voice from a recording (take it from the client with <see cref="Tasks.ITaskClient.GetService(Type, object?)"/>,
/// or cast): turns a recording into a <see cref="VoiceProfile"/> once, which later requests pass in <see cref="SpeechSynthesisOptions.VoiceProfile"/>.
/// </summary>
public interface IVoiceCloner
{
    /// <summary>The architecture family of the profiles this model creates and accepts (<see cref="VoiceProfile.Architecture"/>).</summary>
    public string VoiceArchitecture { get; }

    /// <summary>Derives a voice from a recording; runs on a background thread (the threading contract of the task layer).</summary>
    /// <param name="name">The voice's name.</param>
    /// <param name="recording">The recording of the voice (any rate and channel count).</param>
    /// <param name="referenceText">The words spoken in the recording, for models that align them; <see langword="null"/> when unknown.</param>
    /// <param name="cancellationToken">Stops between the encoders.</param>
    /// <exception cref="NotSupportedException">The model was imported without the encoders cloning needs.</exception>
    public Task<VoiceProfile> CreateVoiceAsync(string name, AudioBuffer recording, string? referenceText = null, CancellationToken cancellationToken = default);
}
