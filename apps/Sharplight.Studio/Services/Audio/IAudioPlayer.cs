namespace Sharplight.Studio.Services.Audio;

/// <summary>Plays a WAV file through the system's audio output.</summary>
internal interface IAudioPlayer
{
    /// <summary>Whether audio can be played on this machine (a player program was found).</summary>
    public bool IsAvailable { get; }

    /// <summary>Why <see cref="IsAvailable"/> is false, or what plays the sound.</summary>
    public string Description { get; }

    /// <summary>Plays <paramref name="path"/> and completes when it finished; cancelling stops it.</summary>
    /// <param name="path">A WAV file.</param>
    /// <param name="cancellationToken">Stops playback.</param>
    public Task PlayAsync(string path, CancellationToken cancellationToken);
}
