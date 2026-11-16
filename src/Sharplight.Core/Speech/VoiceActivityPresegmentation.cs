using System.Runtime.CompilerServices;
using Microsoft.Extensions.DependencyInjection;
using Sharplight.Audio;

namespace Sharplight.Speech;

/// <summary>
/// Pre-segmentation of long recordings by a voice activity detector (<see cref="SpeechRecognitionOptions.UseVoiceActivityDetection"/>): the speech segments are grouped into
/// windows of at most <see cref="MaxWindow"/>, each window is transcribed on its own, and the segment and word times are moved back to the recording's timeline.
/// </summary>
internal static class VoiceActivityPresegmentation
{
    /// <summary>The longest window a recognizer takes in one pass.</summary>
    public static readonly TimeSpan MaxWindow = TimeSpan.FromSeconds(30);

    /// <summary>Transcribes the speech of <paramref name="audio"/> window by window with <paramref name="recognizer"/>.</summary>
    /// <param name="recognizer">The recognizer; called with the detection option off for each window.</param>
    /// <param name="services">The container a detector is resolved from when the options carry none, or <see langword="null"/>.</param>
    /// <param name="audio">The recording.</param>
    /// <param name="options">The request, with <see cref="SpeechRecognitionOptions.UseVoiceActivityDetection"/> on.</param>
    /// <param name="cancellationToken">Stops between windows and inside them.</param>
    /// <exception cref="InvalidOperationException">No detector is set.</exception>
    public static async IAsyncEnumerable<TranscriptionSegment> TranscribeAsync(ISpeechRecognizer recognizer, IServiceProvider? services, AudioBuffer audio, SpeechRecognitionOptions options,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var detector = options.VoiceActivityDetector ?? Resolve(services, options.VoiceActivityDetectorKey)
            ?? throw new InvalidOperationException("SpeechRecognitionOptions.UseVoiceActivityDetection needs a voice activity detector: set SpeechRecognitionOptions.VoiceActivityDetector, "
                + "or register one with services.AddTaskClient<IVoiceActivityDetector>(...) (keyed by SpeechRecognitionOptions.VoiceActivityDetectorKey, or unkeyed).");
        var speech = await detector.DetectAsync(audio, options.VoiceActivity, cancellationToken).ConfigureAwait(false);
        var inner = options with { UseVoiceActivityDetection = false, VoiceActivityDetector = null, VoiceActivityDetectorKey = null };
        foreach (var (start, end) in Windows(speech, audio.SampleRate))
        {
            cancellationToken.ThrowIfCancellationRequested();
            var offset = TimeSpan.FromSeconds((double)start / audio.SampleRate);
            await foreach (var segment in recognizer.TranscribeStreamingAsync(Slice(audio, start, end), inner, cancellationToken).ConfigureAwait(false))
            {
                yield return segment with
                {
                    Start = segment.Start + offset,
                    End = segment.End + offset,
                    Words = [.. segment.Words.Select(w => w with { Start = w.Start + offset, End = w.End + offset })],
                };
            }
        }
    }

    private static IVoiceActivityDetector? Resolve(IServiceProvider? services, object? key)
    {
        if (services is null)
        {
            return null;
        }

        return key is null
            ? services.GetService<IVoiceActivityDetector>()
            : services.GetKeyedService<IVoiceActivityDetector>(key);
    }

    /// <summary>Groups consecutive speech segments into windows of at most <see cref="MaxWindow"/> (a single longer segment is a window of its own).</summary>
    /// <param name="speech">The segments in time order.</param>
    /// <param name="sampleRate">The sample rate of the segment positions.</param>
    internal static IEnumerable<(long Start, long End)> Windows(IReadOnlyList<SpeechSegment> speech, int sampleRate)
    {
        long limit = (long)(MaxWindow.TotalSeconds * sampleRate);
        long start = -1;
        long end = 0;
        foreach (var segment in speech)
        {
            if (start >= 0 && segment.EndSample - start > limit)
            {
                yield return (start, end);
                start = -1;
            }

            if (start < 0)
            {
                start = segment.StartSample;
            }

            end = segment.EndSample;
        }

        if (start >= 0)
        {
            yield return (start, end);
        }
    }

    private static AudioBuffer Slice(AudioBuffer audio, long start, long end)
    {
        int channels = audio.Channels;
        long first = Math.Clamp(start, 0, audio.FrameCount);
        long last = Math.Clamp(end, first, audio.FrameCount);
        return new AudioBuffer(audio.Samples.AsSpan((int)(first * channels), (int)((last - first) * channels)).ToArray(), audio.SampleRate, channels);
    }
}
