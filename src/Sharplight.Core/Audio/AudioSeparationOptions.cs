namespace Sharplight.Audio;

/// <summary>Per-request settings of <see cref="IAudioSeparator"/>.</summary>
public sealed class AudioSeparationOptions
{
    /// <summary>The fraction of a segment shared with the next one (0 to under 1); the overlaps are cross-faded. Default 0.25.</summary>
    public float Overlap { get; set; } = 0.25f;
}
