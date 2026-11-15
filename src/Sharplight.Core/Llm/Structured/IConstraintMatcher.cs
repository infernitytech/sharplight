namespace Sharplight.Llm.Structured;

/// <summary>
/// The per-sequence state of an <see cref="IOutputConstraint"/>. Before each sampled token the engine asks for the allowed
/// next tokens (<see cref="FillAllowed"/>), masks the logits with them, samples, and reports the choice
/// (<see cref="Accept"/>). End-of-generation tokens are the engine's business: it allows them exactly when
/// <see cref="IsComplete"/>. Not thread-safe.
/// </summary>
public interface IConstraintMatcher
{
    /// <summary>True when the text so far is a complete match, so generation may end here.</summary>
    public bool IsComplete { get; }

    /// <summary>True when no further token can extend the match (generation must end).</summary>
    public bool IsTerminated { get; }

    /// <summary>Writes the set of tokens that keep the output a prefix of some match into <paramref name="mask"/> (cleared first).</summary>
    public void FillAllowed(TokenMask mask);

    /// <summary>Advances over <paramref name="token"/>; returns false (state unchanged) when the token is not allowed.</summary>
    public bool Accept(int token);

    /// <summary>Undoes the last <paramref name="count"/> accepted tokens (speculative decoding rolls back rejected drafts).</summary>
    /// <exception cref="ArgumentOutOfRangeException">More tokens than were accepted.</exception>
    public void Rollback(int count);
}
