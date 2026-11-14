using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// Settings of <see cref="CompressedKvAttention"/>, read from the <c>{arch}.attention.compression.*</c> keys: <c>ratios</c> (int[layer_count]:
/// positions per compressed entry of every layer, 0 = the sliding window alone), <c>window</c> (keys of the last that many positions are attended
/// uncompressed), <c>top_k</c> (entries a sparse layer's indexer selects), <c>output_groups</c> (groups of the low-rank output projection) and
/// <c>rope.base_compressed</c> (rope base of the layers with compressed entries; the sliding-only layers use <c>rope.base</c>).
/// </summary>
/// <param name="Ratios">Positions per compressed entry of every layer; 0 for a layer that only attends its window.</param>
/// <param name="Window">Uncompressed keys a query attends (its own position and the ones before it).</param>
/// <param name="TopK">Compressed entries a layer with an indexer selects per query.</param>
/// <param name="OutputGroups">Groups of the low-rank output projection.</param>
/// <param name="CompressedRopeBase">Rope base of the layers with compressed entries.</param>
internal sealed record CompressedKvAttentionOptions(int[] Ratios, int Window, int TopK, int OutputGroups, float CompressedRopeBase)
{
    /// <summary>Reads the options of <paramref name="src"/>; null when the model has no compressed-KV attention.</summary>
    /// <exception cref="InvalidDataException">The ratios do not cover every layer or the window is missing.</exception>
    public static CompressedKvAttentionOptions? Read(ModelSource src, int layers)
    {
        int[] ratios = src.Arch("attention.compression.ratios", Array.Empty<int>());
        if (ratios.Length == 0)
        {
            return null;
        }

        var options = new CompressedKvAttentionOptions(
            ratios,
            src.Arch("attention.compression.window", 0),
            src.Arch("attention.compression.top_k", 0),
            Math.Max(1, src.Arch("attention.compression.output_groups", 1)),
            src.Arch("rope.base_compressed", src.Arch("rope.base", 10000f)));
        if (ratios.Length != layers || ratios.Any(r => r < 0 || r == 1) || options.Window <= 0)
        {
            throw new InvalidDataException("latent_decoder: attention.compression.ratios needs layer_count entries (0 or at least 2) and a positive attention.compression.window.");
        }

        return options;
    }
}
