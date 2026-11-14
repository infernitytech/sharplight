using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>The states of one <see cref="CompressedKvAttention"/> layer.</summary>
/// <param name="Keys">The uncompressed key (= value) cache [ctx, 1, D].</param>
/// <param name="Entries">The compressed cache [ceil(ctx / r), 1, D]; null for a layer that only attends its window.</param>
/// <param name="Pool">The ring of the compressor's last rows (<c>WindowPool</c> state); null without compression or for one position per entry.</param>
/// <param name="IndexEntries">The indexer's compressed key cache; null without an indexer.</param>
/// <param name="IndexPool">The ring of the indexer's own compressor; null without one (also when the index keys derive from the entries).</param>
/// <param name="Borrowed">The compressed caches belong to an earlier layer (its kv source), which writes them; this layer only reads them.</param>
internal sealed record CompressedKvLayerStates(
    StateDefinition Keys, StateDefinition? Entries, StateDefinition? Pool, StateDefinition? IndexEntries, StateDefinition? IndexPool, bool Borrowed = false)
{
    /// <summary>The states the layer owns, in declaration order.</summary>
    public IEnumerable<StateDefinition> All()
    {
        yield return Keys;
        if (Borrowed)
        {
            yield break;
        }

        foreach (var s in new[] { Entries, Pool, IndexEntries, IndexPool })
        {
            if (s is not null)
            {
                yield return s;
            }
        }
    }
}
