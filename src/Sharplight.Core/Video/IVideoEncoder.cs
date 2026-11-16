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

    /// <summary>Encodes the clip. The CPU work runs on the thread pool.</summary>
    /// <param name="clip">The clip.</param>
    /// <param name="output">The destination stream; written sequentially and not disposed.</param>
    /// <param name="options">Encode settings; <see langword="null"/> uses the defaults of <see cref="VideoEncodeOptions"/>.</param>
    /// <param name="cancellationToken">Cancels the encode.</param>
    /// <exception cref="PlatformNotSupportedException"><see cref="IsAvailable"/> is false.</exception>
    public Task EncodeAsync(VideoClip clip, Stream output, VideoEncodeOptions? options = null, CancellationToken cancellationToken = default);
}
