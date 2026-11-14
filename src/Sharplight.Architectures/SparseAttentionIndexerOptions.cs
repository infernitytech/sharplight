using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// Settings of <see cref="SparseAttentionIndexer"/>, read from the <c>{arch}.attention.indexer.*</c> keys: <c>token_budget</c> (top-k, 0 = no
/// indexer), <c>rope_style</c> ("neox" rotate-half, the default, or "gptj" interleaved pairs), <c>rope_last</c> (the rotary part is the end of
/// each query and key instead of its start; Hunyuan v4) and <c>norm_eps</c> (epsilon of the key LayerNorm, default 1e-6).
/// </summary>
/// <param name="TokenBudget">Key positions every query attends (top-k).</param>
/// <param name="RopeStyle">Rotary pairing of the indexer's rope dimensions.</param>
/// <param name="RopeLast">Whether the rotary dimensions are the last ones of each query and key.</param>
/// <param name="NormEps">Epsilon of the key LayerNorm.</param>
internal sealed record SparseAttentionIndexerOptions(int TokenBudget, string RopeStyle, bool RopeLast, float NormEps)
{
    /// <summary>Reads the options of <paramref name="src"/>; null when the model has no indexer.</summary>
    public static SparseAttentionIndexerOptions? Read(ModelSource src)
    {
        int budget = src.Arch("attention.indexer.token_budget", 0);
        return budget <= 0 ? null : new SparseAttentionIndexerOptions(
            budget,
            src.Arch("attention.indexer.rope_style", "neox"),
            src.Arch("attention.indexer.rope_last", false),
            src.Arch("attention.indexer.norm_eps", 1e-6f));
    }
}
