namespace Sharplight.Video.Ffmpeg;

/// <summary>Everything <see cref="FfmpegVideoWriter"/> needs, resolved from the public options.</summary>
internal sealed record FfmpegEncodePlan
{
    public required FfmpegContainer Container { get; init; }

    public required FfmpegCodec Codec { get; init; }

    /// <summary>The FFmpeg pixel format name of the encoded video.</summary>
    public required string PixelFormat { get; init; }

    /// <summary>The encoded size (the clip's size, padded to even values when the codec needs it).</summary>
    public required int Width { get; init; }

    public required int Height { get; init; }

    public required int RateNumerator { get; init; }

    public required int RateDenominator { get; init; }

    /// <summary>The private encoder options (<c>crf</c>, <c>preset</c>, ...).</summary>
    public required IReadOnlyDictionary<string, string> EncoderOptions { get; init; }

    /// <summary>The muxer options (<c>movflags</c>, <c>loop</c>, ...).</summary>
    public required IReadOnlyDictionary<string, string> MuxerOptions { get; init; }

    public int? KeyframeInterval { get; init; }

    public int Threads { get; init; }

    /// <summary>Bits per second, or 0 for none.</summary>
    public long BitRate { get; init; }

    public int? QuantizerMinimum { get; init; }

    public int? QuantizerMaximum { get; init; }

    /// <summary>VideoToolbox's 0-100 quality.</summary>
    public int? GlobalQualityPercent { get; init; }

    public int? CompressionLevel { get; init; }

    /// <summary>The <c>paletteuse</c> dithering of GIF.</summary>
    public string Dither { get; init; } = "sierra2_4a";

    public FfmpegAudioPlan? Audio { get; init; }
}
