namespace Sharplight.Llm;

/// <summary>Sampling configuration. Temperature 0 = greedy.</summary>
public sealed record SamplingOptions
{
    public float Temperature { get; init; } = 0.7f;
    public int TopK { get; init; } = 40;
    public float TopP { get; init; } = 0.95f;
    public float MinP { get; init; } = 0.0f;
    public float RepetitionPenalty { get; init; } = 1.0f;
    public int RepetitionWindow { get; init; } = 64;
    public int? Seed { get; init; }

    public static SamplingOptions Greedy { get; } = new() { Temperature = 0 };
}
