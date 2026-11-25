namespace Sharplight.Audio;

/// <summary>Per-request settings of <see cref="IMusicGenerator"/>; <see langword="null"/> values take the model's defaults (<see cref="IMusicGenerator.Defaults"/>).</summary>
public sealed record MusicGenerationOptions
{
    /// <summary>Classifier-free guidance scale (1 disables guidance), or <see langword="null"/> for the model's default.</summary>
    public float? GuidanceScale { get; init; }

    /// <summary>True for greedy decoding (deterministic; the form that is checked token by token against the reference), false to sample.</summary>
    public bool Greedy { get; init; }

    /// <summary>The sampling temperature, or <see langword="null"/> for the model's default.</summary>
    public float? Temperature { get; init; }

    /// <summary>The sampling keeps the best <c>TopK</c> codes (0 for all of them), or <see langword="null"/> for the model's default.</summary>
    public int? TopK { get; init; }

    /// <summary>The sampling keeps the smallest set of codes whose probability reaches <c>TopP</c> (1 for all of them), or <see langword="null"/> for the model's default.</summary>
    public float? TopP { get; init; }

    /// <summary>
    /// The seed of the sampling; <see langword="null"/> picks one (reported in <see cref="AudioGenerationResult.Seed"/>). Seeds within the <see cref="int"/> range
    /// reproduce the runs of earlier releases.
    /// </summary>
    public long? Seed { get; init; }

    /// <summary>The defaults.</summary>
    public static MusicGenerationOptions Default { get; } = new();
}
