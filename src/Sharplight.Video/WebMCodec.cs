namespace Sharplight.Video;

/// <summary>The video codec inside a WebM file.</summary>
public enum WebMCodec
{
    /// <summary>VP9: better compression, slower encoding.</summary>
    Vp9,

    /// <summary>VP8: faster encoding, larger files, the widest decoder support.</summary>
    Vp8,
}
