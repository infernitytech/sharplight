using Sharplight.Graph;

namespace Sharplight.Architectures;

/// <summary>
/// The values of one graph that later <see cref="CompressedKvAttention"/> layers reuse: the indexer selection of every index source layer and
/// the candidate blocks of the candidate source layer (DeepSeek-V4.1).
/// </summary>
internal sealed class CompressedKvShared
{
    /// <summary>The entries selected by the indexer of a layer, by layer.</summary>
    public Dictionary<int, Value> Selections { get; } = [];

    /// <summary>The candidate blocks (positions of the picked blocks, a group selection's output) of the candidate source layer.</summary>
    public Value? Candidates { get; set; }
}
