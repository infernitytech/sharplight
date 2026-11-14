using Sharplight.Llm.Serving.Speculative;

namespace Sharplight.Families.Tests;

/// <summary>Forwards to another speculator and records every proposal (the sequence's context and the drafts proposed after it).</summary>
internal sealed class RecordingSpeculator(ISpeculator inner) : ISpeculator
{
    /// <summary>The proposals in order: the context the drafts continue, and the drafts.</summary>
    public List<(int[] Context, int[] Drafts)> Proposals { get; } = [];

    /// <inheritdoc/>
    public int MaxDraftTokens => inner.MaxDraftTokens;

    /// <inheritdoc/>
    public bool UsesHiddenStates => inner.UsesHiddenStates;

    /// <inheritdoc/>
    public void Observe(int slot, int position, ReadOnlySpan<int> successors, ReadOnlySpan<float> hidden)
        => inner.Observe(slot, position, successors, hidden);

    /// <inheritdoc/>
    public void Propose(IReadOnlyList<SpeculationRequest> requests)
    {
        inner.Propose(requests);
        foreach (var r in requests)
        {
            if (r.Count > 0)
            {
                Proposals.Add(([.. r.Context], r.Draft[..r.Count]));
            }
        }
    }

    /// <inheritdoc/>
    public void Release(int slot) => inner.Release(slot);

    /// <inheritdoc/>
    public void Dispose() => inner.Dispose();
}
