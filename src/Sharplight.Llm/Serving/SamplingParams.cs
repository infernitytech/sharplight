namespace Sharplight.Llm.Serving;

/// <summary>Per-request sampling configuration. <see cref="Temperature"/> 0 (or <see cref="Greedy"/>) selects the arg-max token.</summary>
public sealed record SamplingParams
{
    public float Temperature { get; init; } = 1f;
    /// <summary>Keep only the K most likely tokens (0 = off).</summary>
    public int TopK { get; init; }
    /// <summary>Nucleus sampling threshold (1 = off).</summary>
    public float TopP { get; init; } = 1f;
    /// <summary>Drop tokens less likely than MinP times the most likely one (0 = off).</summary>
    public float MinP { get; init; }
    /// <summary>Multiplicative (llama.cpp style) penalty on tokens seen in the last <see cref="RepetitionWindow"/> prompt/output tokens (1 = off).</summary>
    public float RepetitionPenalty { get; init; } = 1f;
    public int RepetitionWindow { get; init; } = 256;
    /// <summary>OpenAI style: subtract a flat amount from every token already generated.</summary>
    public float PresencePenalty { get; init; }
    /// <summary>OpenAI style: subtract this times the number of occurrences in the generated text.</summary>
    public float FrequencyPenalty { get; init; }
    public IReadOnlyDictionary<int, float>? LogitBias { get; init; }
    public int? Seed { get; init; }
    public bool Greedy { get; init; }

    public bool IsGreedy => Greedy || Temperature <= 0;
    public static SamplingParams GreedyDefault { get; } = new() { Greedy = true };
}
