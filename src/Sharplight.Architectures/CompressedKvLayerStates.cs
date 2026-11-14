using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>The states of one <see cref="CompressedKvAttention"/> layer.</summary>
/// <param name="Keys">The uncompressed key (= value) cache [ctx, 1, D].</param>
/// <param name="Entries">The compressed cache [ceil(ctx / r), 1, D]; null for a layer that only attends its window.</param>
/// <param name="Pool">The ring of the compressor's last rows (<c>WindowPool</c> state); null without compression.</param>
/// <param name="IndexEntries">The indexer's compressed key cache; null without an indexer.</param>
/// <param name="IndexPool">The ring of the indexer compressor; null without an indexer.</param>
internal sealed record CompressedKvLayerStates(
    StateDefinition Keys, StateDefinition? Entries, StateDefinition? Pool, StateDefinition? IndexEntries, StateDefinition? IndexPool)
{
    /// <summary>The states in declaration order.</summary>
    public IEnumerable<StateDefinition> All()
    {
        yield return Keys;
        foreach (var s in new[] { Entries, Pool, IndexEntries, IndexPool })
        {
            if (s is not null)
            {
                yield return s;
            }
        }
    }
}
