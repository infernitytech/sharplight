namespace Sharplight.Video;

/// <summary>The speed presets the bundled encoders map to their codec's own settings (<see cref="VideoEncodeOptions.Preset"/>).</summary>
public static class VideoPresets
{
    /// <summary>The fastest encode, the largest file for a given quality.</summary>
    public const string Fastest = "fastest";

    /// <summary>Fast.</summary>
    public const string Fast = "fast";

    /// <summary>Balanced.</summary>
    public const string Medium = "medium";

    /// <summary>Slow, smaller files.</summary>
    public const string Slow = "slow";

    /// <summary>The slowest encode, the smallest file.</summary>
    public const string Slowest = "slowest";
}
