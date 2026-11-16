namespace Sharplight.Video;

/// <summary>Settings of a video encode; encoders ignore the settings that do not apply to their format.</summary>
public sealed record VideoEncodeOptions
{
    /// <summary>
    /// Visual quality from 0 (smallest file) to 100 (best): the quantizer of WebM and MP4, the dithering of GIF (100 turns it off) and the
    /// compression effort of APNG (lossless). Ignored by the bitrate-controlled formats when <see cref="BitrateKilobitsPerSecond"/> is set.
    /// </summary>
    public int Quality { get; init; } = 80;

    /// <summary>Target bitrate in kilobit/s for WebM and MP4; <see langword="null"/> selects constant quality from <see cref="Quality"/>.</summary>
    public int? BitrateKilobitsPerSecond { get; init; }

    /// <summary>Whether animated formats (GIF, APNG) repeat forever; false plays them once.</summary>
    public bool Loop { get; init; } = true;

    /// <summary>Worker threads of the encoder; 0 chooses from the processor count.</summary>
    public int MaxThreads { get; init; }
}
