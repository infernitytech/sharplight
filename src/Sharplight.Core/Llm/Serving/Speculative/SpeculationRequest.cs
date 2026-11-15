namespace Sharplight.Llm.Serving.Speculative;

/// <summary>One sequence asking an <see cref="ISpeculator"/> for draft tokens.</summary>
public sealed class SpeculationRequest
{
    /// <summary>The engine slot of the sequence (stable while it runs; speculators may keep per-slot state).</summary>
    public int Slot { get; internal set; }

    /// <summary>
    /// The sequence's tokens so far: prompt, generated tokens and, last, the sampled token that the next forward pass feeds.
    /// Drafts continue after it.
    /// </summary>
    public IReadOnlyList<int> Context { get; internal set; } = [];

    /// <summary>How many draft tokens may be written.</summary>
    public int MaxTokens { get; internal set; }

    /// <summary>Receives the draft tokens (capacity at least <see cref="MaxTokens"/>).</summary>
    public int[] Draft { get; internal set; } = [];

    /// <summary>Set by the speculator: number of tokens written to <see cref="Draft"/> (0 = no proposal).</summary>
    public int Count { get; set; }
}
