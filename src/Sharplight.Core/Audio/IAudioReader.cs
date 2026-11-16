namespace Sharplight.Audio;

/// <summary>Reads decoded audio in chunks (32-bit float, interleaved). Not thread safe; dispose to release the file or memory.</summary>
public interface IAudioReader : IDisposable
{
    /// <summary>The stored format; the reader delivers samples at this rate and channel count.</summary>
    public AudioStreamInfo Info { get; }

    /// <summary>Reads up to <c>destination.Length / Info.Channels</c> frames.</summary>
    /// <param name="destination">Receives interleaved samples.</param>
    /// <returns>The number of frames read; zero at the end of the stream.</returns>
    /// <exception cref="InvalidDataException">The data is corrupt.</exception>
    public int Read(Span<float> destination);

    /// <summary>Moves to a frame position.</summary>
    /// <param name="frame">Zero-based frame index.</param>
    /// <exception cref="NotSupportedException">The format cannot seek.</exception>
    public void Seek(long frame);
}
