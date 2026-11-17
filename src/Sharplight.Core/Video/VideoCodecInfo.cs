namespace Sharplight.Video;

/// <summary>A video codec an <see cref="IVideoEncoder"/> can write, with the choices it offers (for option pickers).</summary>
public sealed record VideoCodecInfo
{
    /// <summary>The id to pass as <see cref="VideoEncodeOptions.Codec"/>.</summary>
    public required string Id { get; init; }

    /// <summary>A short display name.</summary>
    public required string DisplayName { get; init; }

    /// <summary>A one-line description (what the codec is good for, what plays it).</summary>
    public string Description { get; init; } = string.Empty;

    /// <summary>Whether the codec runs on a hardware encoder (GPU / media engine) of this machine.</summary>
    public bool IsHardware { get; init; }

    /// <summary>Whether the codec is lossless (<see cref="VideoEncodeOptions.Quality"/> only affects the compression effort).</summary>
    public bool IsLossless { get; init; }

    /// <summary>The ids for <see cref="VideoEncodeOptions.Preset"/>, fastest first; empty when the codec has none.</summary>
    public IReadOnlyList<string> Presets { get; init; } = [];

    /// <summary>The preset used when <see cref="VideoEncodeOptions.Preset"/> is <see langword="null"/>.</summary>
    public string? DefaultPreset { get; init; }

    /// <summary>The ids for <see cref="VideoEncodeOptions.PixelFormat"/>, the default first.</summary>
    public IReadOnlyList<string> PixelFormats { get; init; } = [];

    /// <summary>The audio codec ids the format accepts next to this video codec, the default first; empty without audio support.</summary>
    public IReadOnlyList<string> AudioCodecs { get; init; } = [];

    /// <summary>The range of <see cref="VideoEncodeOptions.ConstantRateFactor"/> (inclusive), or <see langword="null"/> when the codec has none.</summary>
    public (int Minimum, int Maximum)? ConstantRateFactorRange { get; init; }
}
