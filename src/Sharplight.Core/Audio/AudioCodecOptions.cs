namespace Sharplight.Audio;

/// <summary>Settings of one <see cref="IAudioCodec.EncodeAsync"/> request.</summary>
public sealed class AudioCodecOptions
{
    /// <summary>
    /// How many codebooks to keep (the bit rate: each adds <c>log2(codebook size) · frame rate</c> bits per second); <see langword="null"/> for the codec's
    /// default. Clamped to the codec's <see cref="IAudioCodec.CodebookCount"/>.
    /// </summary>
    public int? Codebooks { get; set; }
}
