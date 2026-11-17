using Sharplight.Audio;
using Sharplight.Studio.Services.Settings;

namespace Sharplight.Studio.Services.Audio;

/// <summary>Plays audio files (WAV, FLAC, MP3, Ogg Vorbis, Ogg Opus) through Sharplight.Audio on the output device chosen in the settings.</summary>
internal sealed class EngineAudioPlayer(IAudioDecoder decoder, IAudioDeviceProvider devices, ISettingsService settings) : IAudioPlayer
{
    /// <inheritdoc/>
    public bool IsAvailable => devices.IsAvailable && devices.GetDevices(AudioDeviceKind.Playback).Count > 0;

    /// <inheritdoc/>
    public string Description => IsAvailable
        ? $"Plays through {devices.Description}."
        : $"No audio output found ({devices.Description}); the audio is saved and can be opened with the system player.";

    /// <inheritdoc/>
    public async Task PlayAsync(string path, CancellationToken cancellationToken)
    {
        if (!IsAvailable)
        {
            throw new NotSupportedException(Description);
        }

        AudioBuffer audio = await Task.Run(() => decoder.Decode(path), cancellationToken).ConfigureAwait(false);
        var options = new AudioStreamOptions
        {
            SampleRate = audio.SampleRate,
            Channels = Math.Min(audio.Channels, 2),
            DeviceName = AudioDeviceChoice.Resolve(devices, AudioDeviceKind.Playback, settings.Current.AudioOutputDevice),
        };
        using IAudioOutput output = devices.OpenOutput(options);
        await output.PlayAsync(audio, cancellationToken).ConfigureAwait(false);
    }

    /// <inheritdoc/>
    public async Task PlayStreamAsync(IAsyncEnumerable<AudioBuffer> audio, CancellationToken cancellationToken)
    {
        if (!IsAvailable)
        {
            throw new NotSupportedException(Description);
        }

        IAudioOutput? output = null;
        try
        {
            await foreach (AudioBuffer buffer in audio.WithCancellation(cancellationToken).ConfigureAwait(false))
            {
                output ??= devices.OpenOutput(new AudioStreamOptions
                {
                    SampleRate = buffer.SampleRate,
                    Channels = Math.Min(buffer.Channels, 2),
                    DeviceName = AudioDeviceChoice.Resolve(devices, AudioDeviceKind.Playback, settings.Current.AudioOutputDevice),
                });
                await output.QueueAsync(buffer, cancellationToken).ConfigureAwait(false);
            }

            if (output is not null)
            {
                await output.DrainAsync(cancellationToken).ConfigureAwait(false);
            }
        }
        finally
        {
            output?.Dispose();
        }
    }
}
