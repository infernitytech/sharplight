using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// Settings of <see cref="CompressedKvAttention"/>, read from the <c>{arch}.attention.compression.*</c> keys: <c>ratios</c> (int[layer_count]:
/// positions per compressed entry of every layer, 0 = the sliding window alone), <c>window</c> (keys of the last that many positions are attended
/// uncompressed), <c>top_k</c> (entries a sparse layer's indexer selects), <c>output_groups</c> (groups of the low-rank output projection) and
/// <c>rope.base_compressed</c> (rope base of the layers with compressed entries; the sliding-only layers use <c>rope.base</c>). Cross-layer
/// sharing (DeepSeek-V4.1): <c>kv_sources</c> (int[layer_count]: the layer whose compressed caches a layer attends, itself by default) and
/// <c>index_sources</c> (the layer whose indexer selection it uses, itself by default); two-level selection: <c>candidate_source</c> (the layer
/// whose indexer also picks <c>candidate_blocks</c> blocks of <c>candidate_block_size</c> entries, -1 = none) inside which the later indexers
/// select. <c>query_norm</c> (default true): every query head is RMS-normalized (without a weight) before the rotation.
/// </summary>
/// <param name="Ratios">Positions per compressed entry of every layer; 0 for a layer that only attends its window.</param>
/// <param name="Window">Uncompressed keys a query attends (its own position and the ones before it).</param>
/// <param name="TopK">Compressed entries a layer with an indexer selects per query.</param>
/// <param name="OutputGroups">Groups of the low-rank output projection.</param>
/// <param name="CompressedRopeBase">Rope base of the layers with compressed entries.</param>
/// <param name="KvSources">Per layer the layer whose compressed caches it attends.</param>
/// <param name="IndexSources">Per layer the layer whose indexer selects its entries.</param>
/// <param name="CandidateSource">The layer whose indexer picks the candidate blocks of the later indexers; -1 without.</param>
/// <param name="CandidateBlocks">Candidate blocks per query.</param>
/// <param name="CandidateBlockSize">Entries per candidate block.</param>
/// <param name="QueryNorm">Every query head is RMS-normalized before the rotation.</param>
internal sealed record CompressedKvAttentionOptions(int[] Ratios, int Window, int TopK, int OutputGroups, float CompressedRopeBase, int[] KvSources,
    int[] IndexSources, int CandidateSource, int CandidateBlocks, int CandidateBlockSize, bool QueryNorm)
{
    /// <summary>Reads the options of <paramref name="src"/>; null when the model has no compressed-KV attention.</summary>
    /// <exception cref="InvalidDataException">The ratios do not cover every layer, the window is missing or a source layer is invalid.</exception>
    public static CompressedKvAttentionOptions? Read(ModelSource src, int layers)
    {
        int[] ratios = src.Arch("attention.compression.ratios", Array.Empty<int>());
        if (ratios.Length == 0)
        {
            return null;
        }

        int[] identity = [.. Enumerable.Range(0, layers)];
        var options = new CompressedKvAttentionOptions(
            ratios,
            src.Arch("attention.compression.window", 0),
            src.Arch("attention.compression.top_k", 0),
            Math.Max(1, src.Arch("attention.compression.output_groups", 1)),
            src.Arch("rope.base_compressed", src.Arch("rope.base", 10000f)),
            src.Arch("attention.compression.kv_sources", identity),
            src.Arch("attention.compression.index_sources", identity),
            src.Arch("attention.compression.candidate_source", -1),
            src.Arch("attention.compression.candidate_blocks", 0),
            src.Arch("attention.compression.candidate_block_size", 0),
            src.Arch("attention.compression.query_norm", true));
        if (ratios.Length != layers || ratios.Any(r => r < 0) || options.Window <= 0)
        {
            throw new InvalidDataException("latent_decoder: attention.compression.ratios needs layer_count non-negative entries and a positive attention.compression.window.");
        }

        for (int il = 0; il < layers && options.KvSources.Length == layers && options.IndexSources.Length == layers; il++)
        {
            int kv = options.KvSources[il];
            int index = options.IndexSources[il];
            if (ratios[il] > 0 && (kv < 0 || kv > il || ratios[kv] != ratios[il] || index > il || (index >= 0 && ratios[index] != ratios[il])))
            {
                throw new InvalidDataException($"latent_decoder: layer {il} has an invalid compressed-KV or indexer source layer.");
            }
        }

        if (options.KvSources.Length != layers || options.IndexSources.Length != layers
            || (options.CandidateSource >= 0 && (options.CandidateBlocks <= 0 || options.CandidateBlockSize <= 0)))
        {
            throw new InvalidDataException("latent_decoder: attention.compression.kv_sources / index_sources need layer_count entries, candidate blocks a positive count and size.");
        }

        return options;
    }
}
