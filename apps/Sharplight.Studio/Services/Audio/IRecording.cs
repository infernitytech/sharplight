namespace Sharplight.Studio.Services.Audio;

/// <summary>A microphone recording in progress.</summary>
internal interface IRecording : IDisposable
{
    /// <summary>Stops recording and completes the WAV file.</summary>
    public Task StopAsync();
}
