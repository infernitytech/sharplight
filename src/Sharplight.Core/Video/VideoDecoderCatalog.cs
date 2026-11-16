using System.Diagnostics.CodeAnalysis;
using Sharplight.Audio;
using Sharplight.Vision;

namespace Sharplight.Video;

/// <summary>
/// Finds the registered <see cref="IVideoDecoder"/> for a file; resolve it from the service provider. Core registers no decoder of its own:
/// <c>Sharplight.Video</c> adds one (<c>AddSharplightVideo()</c>), so <see cref="IsAvailable"/> is false without it.
/// </summary>
public sealed class VideoDecoderCatalog
{
    private readonly IVideoDecoder[] _decoders;

    /// <summary>Creates the catalog over the registered decoders.</summary>
    /// <param name="decoders">All registered decoders; the first available one handles every file.</param>
    public VideoDecoderCatalog(IEnumerable<IVideoDecoder> decoders)
    {
        ArgumentNullException.ThrowIfNull(decoders);
        _decoders = [.. decoders];
    }

    /// <summary>Whether a decoder is available on this machine.</summary>
    public bool IsAvailable => _decoders.Any(d => d.IsAvailable);

    /// <summary>The file extensions the available decoders recognize (for file pickers).</summary>
    public IReadOnlyList<string> FileExtensions => [.. _decoders.Where(d => d.IsAvailable).SelectMany(d => d.FileExtensions).Distinct(StringComparer.OrdinalIgnoreCase)];

    /// <summary>The first available decoder.</summary>
    /// <param name="decoder">The decoder when one is available.</param>
    /// <returns>Whether one is available.</returns>
    public bool TryGet([NotNullWhen(true)] out IVideoDecoder? decoder)
    {
        decoder = _decoders.FirstOrDefault(d => d.IsAvailable);
        return decoder is not null;
    }

    /// <summary>Decodes a file (or a range of it) into a <see cref="VideoClip"/> with its audio track.</summary>
    /// <param name="path">The video file.</param>
    /// <param name="options">Range, size and frame rate; the format is always <see cref="VideoFrameFormat.Rgb24"/>.</param>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>The clip; its frame rate is <see cref="VideoDecodeOptions.FramesPerSecond"/> or the source's average rate.</returns>
    /// <exception cref="PlatformNotSupportedException">No decoder is available.</exception>
    /// <exception cref="InvalidDataException">The file is not a readable video or has no frames in the range.</exception>
    public async Task<VideoClip> ReadClipAsync(string path, VideoDecodeOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        if (!TryGet(out IVideoDecoder? decoder))
        {
            throw new PlatformNotSupportedException("No video decoder is available (the bundled FFmpeg libraries were not found).");
        }

        VideoDecodeOptions settings = (options ?? new VideoDecodeOptions()) with { Format = VideoFrameFormat.Rgb24 };
        VideoInfo info = await decoder.ProbeAsync(path, cancellationToken).ConfigureAwait(false);
        var frames = new List<ImageInput>();
        await foreach (VideoFrame frame in decoder.ReadFramesAsync(path, settings, cancellationToken).ConfigureAwait(false))
        {
            frames.Add(frame.ToImage());
        }

        if (frames.Count == 0)
        {
            throw new InvalidDataException($"'{path}' has no video frames in the requested range.");
        }

        double rate = settings.FramesPerSecond ?? (info.FramesPerSecond > 0 ? info.FramesPerSecond : 25);
        var clip = new VideoClip(frames, rate);
        if (info.AudioCodec is null)
        {
            return clip;
        }

        AudioBuffer? audio = await decoder.ReadAudioAsync(path, null, cancellationToken).ConfigureAwait(false);
        return clip with { Audio = audio is null ? null : Slice(audio, settings.Start, clip.Duration) };
    }

    private static AudioBuffer Slice(AudioBuffer audio, TimeSpan start, TimeSpan duration)
    {
        long first = Math.Min(audio.FrameCount, (long)Math.Round(start.TotalSeconds * audio.SampleRate));
        long count = Math.Min(audio.FrameCount - first, (long)Math.Round(duration.TotalSeconds * audio.SampleRate));
        if (first == 0 && count == audio.FrameCount)
        {
            return audio;
        }

        float[] samples = audio.Samples.AsSpan((int)(first * audio.Channels), (int)(count * audio.Channels)).ToArray();
        return new AudioBuffer(samples, audio.SampleRate, audio.Channels);
    }
}
