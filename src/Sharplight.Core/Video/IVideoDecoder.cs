using Sharplight.Audio;

namespace Sharplight.Video;

/// <summary>Reads video files: their frames as RGB or YUV pictures with timestamps, and their audio track. Thread safe; each read is independent.</summary>
public interface IVideoDecoder
{
    /// <summary>Whether the decoder can work on this machine (its native library is present).</summary>
    public bool IsAvailable { get; }

    /// <summary>The file extensions the decoder recognizes with the leading dot, lower case (it may read other files too).</summary>
    public IReadOnlyList<string> FileExtensions { get; }

    /// <summary>Reads the stream properties of a file.</summary>
    /// <param name="path">The video file.</param>
    /// <param name="cancellationToken">Cancels the probe.</param>
    /// <returns>The properties of the first video stream (and the first audio stream).</returns>
    /// <exception cref="PlatformNotSupportedException"><see cref="IsAvailable"/> is false.</exception>
    /// <exception cref="InvalidDataException">The file is not a readable video.</exception>
    public Task<VideoInfo> ProbeAsync(string path, CancellationToken cancellationToken = default);

    /// <summary>Decodes the frames of the first video stream in presentation order. Decoding runs on the thread pool.</summary>
    /// <param name="path">The video file.</param>
    /// <param name="options">Range, size and pixel layout; <see langword="null"/> returns every frame as RGB at the source size.</param>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>The frames.</returns>
    /// <exception cref="PlatformNotSupportedException"><see cref="IsAvailable"/> is false.</exception>
    /// <exception cref="InvalidDataException">The file is not a readable video.</exception>
    public IAsyncEnumerable<VideoFrame> ReadFramesAsync(string path, VideoDecodeOptions? options = null, CancellationToken cancellationToken = default);

    /// <summary>Decodes the first audio stream to interleaved float samples.</summary>
    /// <param name="path">The video (or audio) file.</param>
    /// <param name="sampleRate">Resample to this rate in Hz; <see langword="null"/> keeps the source rate.</param>
    /// <param name="cancellationToken">Cancels the read.</param>
    /// <returns>The audio, or <see langword="null"/> when the file has no audio stream.</returns>
    /// <exception cref="PlatformNotSupportedException"><see cref="IsAvailable"/> is false.</exception>
    /// <exception cref="InvalidDataException">The file is not readable.</exception>
    public Task<AudioBuffer?> ReadAudioAsync(string path, int? sampleRate = null, CancellationToken cancellationToken = default);
}
