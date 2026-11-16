using Sharplight.Tasks;

namespace Sharplight.Audio;

/// <summary>
/// A neural audio codec (kind <see cref="Models.ModelKinds.AudioCodec"/>: EnCodec, DAC, Mimi, SNAC, ...): audio in, discrete residual-vector-quantizer codes
/// out, and codes back to audio. Follows the task-client pattern of every model kind (<see cref="ITaskClient"/>). The calls run the compiled graphs off the
/// caller's thread. Not thread-safe: use one codec per concurrent request.
/// </summary>
public interface IAudioCodec : ITaskClient
{
    /// <summary>The sample rate of the audio the codec takes and returns.</summary>
    public int SampleRate { get; }

    /// <summary>The number of audio samples one frame of codes covers.</summary>
    public int SamplesPerFrame { get; }

    /// <summary>The frames per second (<c>SampleRate / SamplesPerFrame</c>).</summary>
    public double FrameRate { get; }

    /// <summary>The number of codebooks the codec has.</summary>
    public int CodebookCount { get; }

    /// <summary>The number of codebooks used when <see cref="AudioCodecOptions.Codebooks"/> is not set.</summary>
    public int DefaultCodebooks { get; }

    /// <summary>The number of entries of each codebook (the codes are in <c>[0, CodebookSize)</c>).</summary>
    public int CodebookSize { get; }

    /// <summary>Encodes audio: it is converted to mono at <see cref="SampleRate"/> first, and zero-padded at the end to a whole number of frames.</summary>
    /// <param name="audio">The audio.</param>
    /// <param name="options">Request settings, or <see langword="null"/> for the defaults.</param>
    /// <param name="cancellationToken">Stops encoding between segments.</param>
    public Task<CodecCodes> EncodeAsync(AudioBuffer audio, AudioCodecOptions? options = null, CancellationToken cancellationToken = default);

    /// <summary>Decodes codes back to mono audio at <see cref="SampleRate"/>; fewer codebooks than <see cref="CodebookCount"/> give a lower-quality reconstruction.</summary>
    /// <param name="codes">The codes.</param>
    /// <param name="cancellationToken">Stops decoding between segments.</param>
    public Task<AudioBuffer> DecodeAsync(CodecCodes codes, CancellationToken cancellationToken = default);
}
