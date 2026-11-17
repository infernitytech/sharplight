namespace Sharplight.Video;

/// <summary>
/// Settings of a video encode; encoders ignore the settings that do not apply to their format. The string settings (<see cref="Codec"/>,
/// <see cref="Preset"/>, <see cref="PixelFormat"/>) are ids the encoder lists in <see cref="IVideoEncoder.Codecs"/>; <see langword="null"/>
/// picks the encoder's default.
/// </summary>
public sealed record VideoEncodeOptions
{
    /// <summary>
    /// Visual quality from 0 (smallest file) to 100 (best): mapped to the codec's quantizer / constant rate factor, the dithering of GIF
    /// (100 turns it off) and the compression effort of the lossless formats. Ignored when <see cref="BitrateKilobitsPerSecond"/> or
    /// <see cref="ConstantRateFactor"/> is set.
    /// </summary>
    public int Quality { get; init; } = 80;

    /// <summary>Target bitrate in kilobit/s; <see langword="null"/> selects constant quality from <see cref="Quality"/>.</summary>
    public int? BitrateKilobitsPerSecond { get; init; }

    /// <summary>
    /// The codec's own constant rate factor or quantizer (its native scale, for example 0-63 for AV1), overriding <see cref="Quality"/>;
    /// <see langword="null"/> derives it from <see cref="Quality"/>.
    /// </summary>
    public int? ConstantRateFactor { get; init; }

    /// <summary>The video codec id (<see cref="VideoCodecInfo.Id"/> of one of the encoder's <see cref="IVideoEncoder.Codecs"/>).</summary>
    public string? Codec { get; init; }

    /// <summary>The speed / compression preset id (one of <see cref="VideoCodecInfo.Presets"/>).</summary>
    public string? Preset { get; init; }

    /// <summary>The pixel format id of the encoded video (one of <see cref="VideoCodecInfo.PixelFormats"/>).</summary>
    public string? PixelFormat { get; init; }

    /// <summary>Plays the frames at this rate instead of the clip's <see cref="VideoClip.FramesPerSecond"/> (positive); no frames are dropped or added.</summary>
    public double? FramesPerSecond { get; init; }

    /// <summary>The longest distance between two key frames, in frames; <see langword="null"/> uses the codec's default.</summary>
    public int? KeyframeInterval { get; init; }

    /// <summary>Whether the clip's <see cref="VideoClip.Audio"/> is written as an audio track where the format supports one.</summary>
    public bool IncludeAudio { get; init; } = true;

    /// <summary>The audio codec id (one of <see cref="VideoCodecInfo.AudioCodecs"/>); <see langword="null"/> uses the format's default.</summary>
    public string? AudioCodec { get; init; }

    /// <summary>Audio bitrate in kilobit/s; <see langword="null"/> uses the codec's default.</summary>
    public int? AudioBitrateKilobitsPerSecond { get; init; }

    /// <summary>Whether animated formats (GIF, APNG) repeat forever; false plays them once.</summary>
    public bool Loop { get; init; } = true;

    /// <summary>Worker threads of the encoder; 0 chooses from the processor count.</summary>
    public int MaxThreads { get; init; }
}
