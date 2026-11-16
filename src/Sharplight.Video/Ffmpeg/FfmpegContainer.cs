namespace Sharplight.Video.Ffmpeg;

/// <summary>One output format of the bundled FFmpeg: the muxer and the codecs it accepts.</summary>
internal sealed record FfmpegContainer
{
    /// <summary>The format id (<see cref="VideoFormats"/>).</summary>
    public required string Format { get; init; }

    /// <summary>The file extensions, the preferred first.</summary>
    public required IReadOnlyList<string> FileExtensions { get; init; }

    /// <summary>The FFmpeg muxer name.</summary>
    public required string Muxer { get; init; }

    /// <summary>The software codecs, the default first; the hardware ones of <see cref="Hardware"/> follow when they work.</summary>
    public required IReadOnlyList<FfmpegCodec> Codecs { get; init; }

    /// <summary>The hardware codecs the format accepts.</summary>
    public IReadOnlyList<FfmpegCodec> Hardware { get; init; } = [];

    /// <summary>The audio codec ids, the default first; empty when the format has no audio.</summary>
    public IReadOnlyList<string> AudioCodecs { get; init; } = [];
}
