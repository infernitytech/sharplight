namespace Sharplight.Video;

/// <summary>Writes a <see cref="VideoClip"/> in one video file format. Implementations are stateless and thread safe.</summary>
public interface IVideoEncoder
{
    /// <summary>The format id, one of <see cref="VideoFormats"/> for the built-in formats.</summary>
    public string Format { get; }

    /// <summary>The file extensions of the format with the leading dot, lower case, the preferred one first (for example <c>.webm</c>).</summary>
    public IReadOnlyList<string> FileExtensions { get; }

    /// <summary>Whether the encoder can work on this machine (its native library or external program is present).</summary>
    public bool IsAvailable { get; }

    /// <summary>
    /// Which encoder wins when several handle a format (<see cref="VideoEncoderCatalog"/> takes the available one with the highest value,
    /// then the first registered). The managed encoders use 0, the bundled native encoders a higher value.
    /// </summary>
    public int Priority => 0;

    /// <summary>The video codecs this encoder can write on this machine, the default first; empty for single-codec formats without choices.</summary>
    public IReadOnlyList<VideoCodecInfo> Codecs => [];

    /// <summary>Encodes the clip. The CPU work runs on the thread pool.</summary>
    /// <param name="clip">The clip.</param>
    /// <param name="output">The destination stream; written sequentially (and seeked when it can seek) and not disposed.</param>
    /// <param name="options">Encode settings; <see langword="null"/> uses the defaults of <see cref="VideoEncodeOptions"/>.</param>
    /// <param name="cancellationToken">Cancels the encode.</param>
    /// <exception cref="PlatformNotSupportedException"><see cref="IsAvailable"/> is false.</exception>
    /// <exception cref="ArgumentException">An option names a codec, preset or pixel format the encoder does not offer.</exception>
    public Task EncodeAsync(VideoClip clip, Stream output, VideoEncodeOptions? options = null, CancellationToken cancellationToken = default);

    /// <summary>Whether <see cref="OpenWriter"/> works: the encoder writes frames as they come, with memory independent of the video's length.</summary>
    public bool SupportsIncrementalWriting => false;

    /// <summary>Starts an incremental encode: frames are written one at a time and the file is finished by <see cref="IVideoFrameWriter.CompleteAsync"/>.</summary>
    /// <param name="output">The destination stream; written sequentially (and seeked when it can seek) and not disposed.</param>
    /// <param name="stream">The frame size and rate and the source of the sound track.</param>
    /// <param name="options">Encode settings; <see langword="null"/> uses the defaults of <see cref="VideoEncodeOptions"/>.</param>
    /// <returns>The writer; dispose it.</returns>
    /// <exception cref="NotSupportedException"><see cref="SupportsIncrementalWriting"/> is false.</exception>
    /// <exception cref="PlatformNotSupportedException"><see cref="IsAvailable"/> is false.</exception>
    /// <exception cref="ArgumentException">An option names a codec, preset or pixel format the encoder does not offer.</exception>
    public IVideoFrameWriter OpenWriter(Stream output, VideoStreamSettings stream, VideoEncodeOptions? options = null)
        => throw new NotSupportedException($"The {Format} encoder writes whole clips only.");
}
