namespace Sharplight.Llm;

/// <summary>Sampling configuration. Temperature 0 = greedy.</summary>
public sealed record SamplingOptions
{
    /// <summary>Softmax temperature; 0 (or less) selects the most likely token.</summary>
    public float Temperature { get; init; } = 0.7f;
    /// <summary>Keep only the K most likely tokens; 0 disables.</summary>
    public int TopK { get; init; } = 40;
    /// <summary>Nucleus sampling: keep the smallest set of tokens whose probability mass reaches this value; 1 disables.</summary>
    public float TopP { get; init; } = 0.95f;
    /// <summary>Drop tokens less likely than this fraction of the most likely one; 0 disables.</summary>
    public float MinP { get; init; } = 0.0f;
    /// <summary>Divides positive (multiplies negative) logits of recently seen tokens by this factor; 1 disables.</summary>
    public float RepetitionPenalty { get; init; } = 1.0f;
    /// <summary>Number of most recent tokens the repetition penalty looks at.</summary>
    public int RepetitionWindow { get; init; } = 64;
    /// <summary>Seed for reproducible sampling; null seeds from entropy.</summary>
    public int? Seed { get; init; }

    /// <summary>Always pick the most likely token.</summary>
    public static SamplingOptions Greedy { get; } = new() { Temperature = 0 };
}
