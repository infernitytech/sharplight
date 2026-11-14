namespace Sharplight.Studio.Models;

/// <summary>How tokens are sampled and how long an answer may get.</summary>
internal sealed record SamplingSettings
{
    /// <summary>Softmax temperature; 0 is greedy.</summary>
    public double Temperature { get; init; } = 0.7;

    /// <summary>Nucleus sampling mass (1 = off).</summary>
    public double TopP { get; init; } = 0.95;

    /// <summary>Keep only the k most likely tokens (0 = off).</summary>
    public int TopK { get; init; } = 40;

    /// <summary>Drop tokens below this fraction of the most likely one (0 = off).</summary>
    public double MinP { get; init; } = 0.05;

    /// <summary>Penalty for recently generated tokens (1 = off).</summary>
    public double RepetitionPenalty { get; init; } = 1.0;

    /// <summary>Maximum generated tokens (0 = until the context is full).</summary>
    public int MaxTokens { get; init; } = 2048;

    /// <summary>Random seed; null for a different one every time.</summary>
    public int? Seed { get; init; }
}
