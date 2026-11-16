namespace Sharplight.Audio;

/// <summary>The discrete codes an <see cref="IAudioCodec"/> produced for a piece of audio: one code per codebook per frame.</summary>
public sealed class CodecCodes
{
    /// <summary>Creates codes from codebook-major values.</summary>
    /// <param name="codebooks">The number of codebooks (residual quantizer stages).</param>
    /// <param name="frames">The number of frames.</param>
    /// <param name="values">The codes, codebook-major: <c>values[codebook * frames + frame]</c>.</param>
    /// <exception cref="ArgumentException">The array length is not <c>codebooks · frames</c>.</exception>
    public CodecCodes(int codebooks, int frames, int[] values)
    {
        ArgumentNullException.ThrowIfNull(values);
        ArgumentOutOfRangeException.ThrowIfNegative(codebooks);
        ArgumentOutOfRangeException.ThrowIfNegative(frames);
        if (values.Length != (long)codebooks * frames)
        {
            throw new ArgumentException($"Expected {(long)codebooks * frames} codes for {codebooks} codebooks and {frames} frames, got {values.Length}.", nameof(values));
        }

        Codebooks = codebooks;
        Frames = frames;
        Values = values;
    }

    /// <summary>The number of codebooks (residual quantizer stages).</summary>
    public int Codebooks { get; }

    /// <summary>The number of frames.</summary>
    public int Frames { get; }

    /// <summary>The codes, codebook-major: <c>Values[codebook * Frames + frame]</c>.</summary>
    public int[] Values { get; }

    /// <summary>The code of one codebook at one frame.</summary>
    /// <param name="codebook">The codebook.</param>
    /// <param name="frame">The frame.</param>
    public int this[int codebook, int frame] => Values[(codebook * Frames) + frame];

    /// <summary>The codes of one codebook over all frames.</summary>
    /// <param name="codebook">The codebook.</param>
    public ReadOnlySpan<int> Codebook(int codebook) => Values.AsSpan(codebook * Frames, Frames);
}
