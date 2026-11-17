namespace Sharplight.Video.Ffmpeg;

/// <summary>How <see cref="VideoEncodeOptions.Quality"/> / <see cref="VideoEncodeOptions.ConstantRateFactor"/> reach an encoder.</summary>
internal enum QualityControl
{
    /// <summary>No quality setting (lossless or profile-controlled codecs).</summary>
    None,

    /// <summary>The encoder's <c>crf</c> option.</summary>
    Crf,

    /// <summary>NVENC's constant quality (<c>cq</c>, variable bitrate without a target).</summary>
    NvencConstantQuality,

    /// <summary>OpenH264: a narrow QP window around the target (it has no CRF).</summary>
    QuantizerWindow,

    /// <summary>A 0-100 quality (VideoToolbox's global quality, Media Foundation's <c>quality</c>).</summary>
    Percent,

    /// <summary>The PNG compression level (0-9) from the quality, for the lossless APNG encoder.</summary>
    CompressionLevel,

    /// <summary>GIF: the palette dithering (none at quality 100).</summary>
    Dither,
}
