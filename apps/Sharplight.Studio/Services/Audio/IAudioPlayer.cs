namespace Sharplight.Studio.Services.Audio;

/// <summary>Plays an audio file through the system's audio output.</summary>
internal interface IAudioPlayer
{
    /// <summary>Whether audio can be played on this machine (an audio output device was found).</summary>
    public bool IsAvailable { get; }

    /// <summary>Why <see cref="IsAvailable"/> is false, or what plays the sound.</summary>
    public string Description { get; }

    /// <summary>Plays <paramref name="path"/> and completes when it finished; cancelling stops it.</summary>
    /// <param name="path">An audio file (WAV, FLAC, MP3, Ogg Vorbis or Ogg Opus).</param>
    /// <param name="cancellationToken">Stops playback.</param>
    public Task PlayAsync(string path, CancellationToken cancellationToken);
}
