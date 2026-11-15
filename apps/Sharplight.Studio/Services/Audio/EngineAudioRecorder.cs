using Sharplight.Audio;
using Sharplight.Studio.Services.Settings;

namespace Sharplight.Studio.Services.Audio;

/// <summary>Records the microphone through Sharplight.Audio (mono, 16 kHz) on the input device chosen in the settings.</summary>
internal sealed class EngineAudioRecorder(IAudioDeviceProvider devices, IAudioEncoder encoder, ISettingsService settings) : IAudioRecorder
{
    private const int SampleRate = 16000;

    /// <inheritdoc/>
    public bool IsAvailable => devices.IsAvailable && devices.GetDevices(AudioDeviceKind.Capture).Count > 0;

    /// <inheritdoc/>
    public string Description => IsAvailable
        ? $"Records through {devices.Description}."
        : $"No microphone found ({devices.Description}); choose an audio file instead.";

    /// <inheritdoc/>
    public IRecording Start(string path)
    {
        if (!IsAvailable)
        {
            throw new NotSupportedException(Description);
        }

        var options = new AudioStreamOptions
        {
            SampleRate = SampleRate,
            Channels = 1,
            BufferDuration = TimeSpan.FromSeconds(2),
            DeviceName = AudioDeviceChoice.Resolve(devices, AudioDeviceKind.Capture, settings.Current.AudioInputDevice),
        };
        IAudioInput input = devices.OpenInput(options);
        try
        {
            input.Start();
        }
        catch
        {
            input.Dispose();
            throw;
        }

        return new CaptureRecording(input, encoder, path);
    }
}
