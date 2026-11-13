namespace Sharplight.Llm.Serving.Speculative;

/// <summary>
/// Proposes draft tokens for speculative decoding. Each engine step, the engine asks for drafts of the decoding sequences
/// (<see cref="Propose"/>), feeds every sequence's pending token plus its drafts to the target model in one forward pass with a
/// logits row per position, keeps the longest prefix of drafts that the target's own sampling reproduces, and rolls the target
/// context back over the rest. Output is therefore distributed exactly as without speculation.
/// <para>
/// Implementations: <see cref="NgramSpeculator"/> (prompt lookup, no extra model) and <see cref="DraftModelSpeculator"/>
/// (a small model of the same family). A multi-token-prediction head of the target model fits the same contract: it proposes
/// from the target's hidden states, which a decoder exposing them could hand to such a speculator.
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
}
