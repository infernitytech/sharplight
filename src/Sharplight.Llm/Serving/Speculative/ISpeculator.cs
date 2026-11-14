namespace Sharplight.Llm.Serving.Speculative;

/// <summary>
/// Proposes draft tokens for speculative decoding. Each engine step, the engine asks for drafts of the decoding sequences
/// (<see cref="Propose"/>), feeds every sequence's pending token plus its drafts to the target model in one forward pass with a
/// logits row per position, keeps the longest prefix of drafts that the target's own sampling reproduces, and rolls the target
/// context back over the rest. Output is therefore distributed exactly as without speculation.
/// <para>
/// Implementations: <see cref="NgramSpeculator"/> (prompt lookup, no extra model) and <see cref="MtpSpeculator"/> (the target
/// model's own multi-token-prediction head, fed the target's hidden states through <see cref="Observe"/>). A small draft model of
/// the same family fits the same contract.
/// </para>
/// <para>
/// Drafts are point proposals: the engine verifies them by sampling each position from the target's own distribution and
/// keeping drafts while they equal the sample, which is speculative rejection sampling for a deterministic draft (accept with
/// probability p(draft), otherwise sample from p restricted to the other tokens), so output is distributed as without
/// speculation for every sampling setting.
/// </para>
/// Called only from the engine's scheduler thread.
/// </summary>
public interface ISpeculator : IDisposable
{
    /// <summary>Most draft tokens proposed per sequence and step.</summary>
    public int MaxDraftTokens { get; }

    /// <summary>
    /// Fills <see cref="SpeculationRequest.Draft"/> of each request with up to <see cref="SpeculationRequest.MaxTokens"/> tokens
    /// that likely follow <see cref="SpeculationRequest.Context"/> and sets <see cref="SpeculationRequest.Count"/>.
    /// </summary>
    public void Propose(IReadOnlyList<SpeculationRequest> requests);

    /// <summary>The sequence in <paramref name="slot"/> ended; drop any per-slot state.</summary>
    public void Release(int slot);

    /// <summary>
    /// Whether the speculator wants the target's hidden states (<see cref="Observe"/>); the engine then has the decoder capture
    /// them (<see cref="IBatchDecoder.CaptureHiddenStates"/>), and speculation needs a decoder that exposes them.
    /// </summary>
    public bool UsesHiddenStates => false;

    /// <summary>
    /// After a step, the rows the target kept for the sequence in <paramref name="slot"/>: positions <paramref name="position"/>
    /// onwards, their hidden states (<paramref name="hidden"/>, rows of <see cref="IBatchDecoder.HiddenSize"/>) and the token that
    /// follows each position (<paramref name="successors"/>; the last one is the token the next step feeds). Called only when
    /// <see cref="UsesHiddenStates"/>, and not for every step: positions may jump (rows the speculator never saw).
    /// </summary>
    public void Observe(int slot, int position, ReadOnlySpan<int> successors, ReadOnlySpan<float> hidden)
    {
    }
}
