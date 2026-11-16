namespace Sharplight.Video.Ffmpeg;

/// <summary>How one public codec id maps to an FFmpeg encoder and its options.</summary>
internal sealed record FfmpegCodec
{
    /// <summary>The public id (<see cref="VideoCodecs"/>).</summary>
    public required string Id { get; init; }

    /// <summary>The display name.</summary>
    public required string DisplayName { get; init; }

    /// <summary>A one-line description.</summary>
    public required string Description { get; init; }

    /// <summary>The FFmpeg encoder name (<c>libsvtav1</c>).</summary>
    public required string EncoderName { get; init; }

    /// <summary>Whether it is a hardware encoder that is only offered after a trial open succeeded.</summary>
    public bool IsHardware { get; init; }

    /// <summary>Whether the codec is lossless.</summary>
    public bool IsLossless { get; init; }

    /// <summary>The offered pixel formats (FFmpeg names), the default first.</summary>
    public required IReadOnlyList<string> PixelFormats { get; init; }

    /// <summary>The preset ids, fastest first, with the encoder options each sets.</summary>
    public IReadOnlyList<(string Id, IReadOnlyDictionary<string, string> Options)> Presets { get; init; } = [];

    /// <summary>The preset used without <see cref="VideoEncodeOptions.Preset"/>.</summary>
    public string? DefaultPreset { get; init; }

    /// <summary>How the quality is applied.</summary>
    public QualityControl Quality { get; init; }

    /// <summary>The rate factor at quality 0 (the codec's worst useful value).</summary>
    public int WorstRateFactor { get; init; }

    /// <summary>The rate factor at quality 100.</summary>
    public int BestRateFactor { get; init; }

    /// <summary>The valid range of <see cref="VideoEncodeOptions.ConstantRateFactor"/>, or <see langword="null"/> without one.</summary>
    public (int Minimum, int Maximum)? RateFactorRange { get; init; }

    /// <summary>Encoder options that are always set.</summary>
    public IReadOnlyDictionary<string, string> FixedOptions { get; init; } = new Dictionary<string, string>();

    /// <summary>Whether frames must have even dimensions (chroma subsampling); odd sizes are padded by one pixel.</summary>
    public bool NeedsEvenSize { get; init; } = true;

    /// <summary>Whether frames go through the palette filter graph (GIF) instead of a plain conversion.</summary>
    public bool UsesPaletteFilter { get; init; }

    /// <summary>Whether a bitrate cap must accompany the rate factor (libvpx VP8's constrained quality mode).</summary>
    public bool NeedsBitrateWithRateFactor { get; init; }

    /// <summary>The public description of the codec.</summary>
    /// <param name="audioCodecs">The audio codec ids of the container.</param>
    /// <returns>The description.</returns>
    public VideoCodecInfo ToInfo(IReadOnlyList<string> audioCodecs) => new()
    {
        Id = Id,
        DisplayName = DisplayName,
        Description = Description,
        IsHardware = IsHardware,
        IsLossless = IsLossless,
        Presets = [.. Presets.Select(p => p.Id)],
        DefaultPreset = DefaultPreset,
        PixelFormats = PixelFormats,
        AudioCodecs = audioCodecs,
        ConstantRateFactorRange = RateFactorRange,
    };
}
