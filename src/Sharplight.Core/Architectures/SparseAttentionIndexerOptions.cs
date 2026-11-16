using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// Settings of <see cref="SparseAttentionIndexer"/>, read from the <c>{arch}.attention.indexer.*</c> keys: <c>token_budget</c> (top-k, 0 = no
/// indexer), <c>rope_style</c> ("neox" rotate-half, the default, or "gptj" interleaved pairs), <c>rope_last</c> (the rotary part is the end of
/// each query and key instead of its start; Hunyuan v4), <c>norm_eps</c> (epsilon of the key LayerNorm, default 1e-6), <c>pool_size</c> (G &gt; 1:
/// the indexer scores pooled keys of G consecutive positions and picks whole pools, GLM-5.3; default 1) and <c>pool_tail</c> (bool: the
/// positions of the query's incomplete pool are attended as well).
/// </summary>
/// <param name="TokenBudget">Key positions every query attends (top-k), a multiple of <paramref name="PoolSize"/>.</param>
/// <param name="RopeStyle">Rotary pairing of the indexer's rope dimensions.</param>
/// <param name="RopeLast">Whether the rotary dimensions are the last ones of each query and key.</param>
/// <param name="NormEps">Epsilon of the key LayerNorm.</param>
/// <param name="PoolSize">Positions per pooled key (1: every position is scored by its own key).</param>
/// <param name="PoolTail">Whether the positions of the query's incomplete pool follow the selection.</param>
internal sealed record SparseAttentionIndexerOptions(int TokenBudget, string RopeStyle, bool RopeLast, float NormEps, int PoolSize = 1, bool PoolTail = false)
{
    /// <summary>Whether keys are pooled in groups of <see cref="PoolSize"/> positions.</summary>
    public bool Pooled => PoolSize > 1;

    /// <summary>Reads the options of <paramref name="src"/>; null when the model has no indexer.</summary>
    /// <exception cref="InvalidDataException">The budget is not a multiple of the pool size.</exception>
    public static SparseAttentionIndexerOptions? Read(ModelSource src)
    {
        int budget = src.Arch("attention.indexer.token_budget", 0);
        if (budget <= 0)
        {
            return null;
        }

        var options = new SparseAttentionIndexerOptions(
            budget,
            src.Arch("attention.indexer.rope_style", "neox"),
            src.Arch("attention.indexer.rope_last", false),
            src.Arch("attention.indexer.norm_eps", 1e-6f),
            Math.Max(1, src.Arch("attention.indexer.pool_size", 1)),
            src.Arch("attention.indexer.pool_tail", false));
        return budget % options.PoolSize == 0
            ? options
            : throw new InvalidDataException($"attention.indexer.token_budget {budget} is not a multiple of attention.indexer.pool_size {options.PoolSize}.");
    }
}
