using Sharplight.Llm.Serving.Speculative;

namespace Sharplight.Serving.Tests;

/// <summary>Test speculator: drafts the most likely continuation according to <paramref name="mostLikely"/>, applied repeatedly.</summary>
public sealed class GreedyGuessSpeculator(int maxDraftTokens, Func<IReadOnlyList<int>, int> mostLikely) : ISpeculator
{
    public int MaxDraftTokens => maxDraftTokens;

    public void Propose(IReadOnlyList<SpeculationRequest> requests)
    {
        foreach (var r in requests)
        {
            var context = r.Context.ToList();
            int n = Math.Min(r.MaxTokens, maxDraftTokens);
            for (int i = 0; i < n; i++)
            {
                r.Draft[i] = mostLikely(context);
                context.Add(r.Draft[i]);
            }

            r.Count = n;
        }
    }

    public void Release(int slot)
    {
    }

    public void Dispose()
    {
    }
}
