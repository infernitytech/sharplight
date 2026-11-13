using Sharplight.Graph;

namespace Sharplight.Shaders.Lowering;

/// <summary>
/// Scatters of a folded ScatterND chain that one gather kernel handles: their update tensors and the distinct buffers
/// those live in (one source pointer each, based at the lowest update offset in that buffer).
/// </summary>
internal sealed class ScatterGatherPass
{
    /// <summary>The scatters of this pass, in execution order.</summary>
    public List<Node> Nodes { get; } = [];

    /// <summary>The update tensor of each scatter.</summary>
    public List<TensorRef> Updates { get; } = [];

    /// <summary>Distinct buffers holding the updates (source slot = index).</summary>
    public List<BufferDecl> Buffers { get; } = [];

    /// <summary>Per source slot, the byte offset its pointer starts at.</summary>
    public List<long> BaseOffsets { get; } = [];

    /// <summary>Adds a scatter and its update tensor.</summary>
    public void Add(Node node, TensorRef update)
    {
        Nodes.Add(node);
        Updates.Add(update);
        int slot = Buffers.IndexOf(update.Buffer);
        if (slot < 0)
        {
            Buffers.Add(update.Buffer);
            BaseOffsets.Add(update.ByteOffset);
        }
        else
        {
            BaseOffsets[slot] = Math.Min(BaseOffsets[slot], update.ByteOffset);
        }
    }
}
