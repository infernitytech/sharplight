namespace Sharplight.Studio.Services.Audio;

/// <summary>Records the microphone to a WAV file.</summary>
internal interface IAudioRecorder
{
    /// <summary>Whether a microphone can be recorded on this machine (a recorder program was found).</summary>
    public bool IsAvailable { get; }

    /// <summary>Why <see cref="IsAvailable"/> is false, or what records the sound.</summary>
    public string Description { get; }

    /// <summary>Starts recording to <paramref name="path"/> (mono, 16 kHz, 16 bit).</summary>
    /// <param name="path">The WAV file to write.</param>
    /// <returns>The running recording; disposing it or <see cref="IRecording.StopAsync"/> finishes the file.</returns>
    public IRecording Start(string path);
}
