namespace Sharplight.Audio;

/// <summary>Convenience members over <see cref="IAudioOutput"/>.</summary>
public static class AudioOutputExtensions
{
    extension(IAudioOutput output)
    {
        /// <summary>
        /// Plays a buffer to the end (converting sample rate and channel count to the stream's) and completes when the last sample was
        /// handed to the device. Starts the stream when it is stopped; cancelling stops the stream and drops queued samples' playback.
        /// </summary>
        /// <param name="audio">The audio to play.</param>
        /// <param name="cancellationToken">Cancels playback.</param>
        public async Task PlayAsync(AudioBuffer audio, CancellationToken cancellationToken = default)
        {
            ArgumentNullException.ThrowIfNull(output);
            ArgumentNullException.ThrowIfNull(audio);
            AudioBuffer converted = audio.ConvertTo(output.SampleRate, output.Channels);
            output.Start();
            try
            {
                int channels = output.Channels;
                int offset = 0;
                while (offset < converted.Samples.Length)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    int accepted = output.Write(converted.Samples.AsSpan(offset));
                    offset += accepted * channels;
                    if (accepted == 0)
                    {
                        await Task.Delay(10, cancellationToken).ConfigureAwait(false);
                    }
                }

                while (output.QueuedFrames > 0)
                {
                    await Task.Delay(10, cancellationToken).ConfigureAwait(false);
                }
            }
            catch (OperationCanceledException)
            {
                output.Stop();
                throw;
            }
        }
    }
}
