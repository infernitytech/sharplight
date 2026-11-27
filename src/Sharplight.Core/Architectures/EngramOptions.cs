using System.Runtime.InteropServices;
using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// Configuration of the per-layer hashed n-gram memory (<see cref="EngramMemory"/>), read from the canonical <c>{arch}.engram.*</c>
/// hyper-parameters: <c>layers</c> (the layers that read the memory, ascending; module e belongs to the e-th of them), <c>order</c> (n-gram
/// length N including the token itself), <c>heads_per_order</c>, <c>multipliers</c> (N per module), <c>head_sizes</c> / <c>head_offsets</c>
/// ((N - 1)·heads_per_order per module: bucket count and first row of every hash head in the shared table), <c>token_map</c> (token id →
/// hashed id; negative: no id), <c>pad_token</c> (the token whose hashed id stands for missing tokens) and <c>norm_eps</c> (epsilon of the
/// module's RMS norms; default the model's). The multipliers and the token map may instead be the weights <c>engram.hash_multipliers</c> (i64) and
/// <c>engram.token_ids</c> (i32 [V]), which an importer derives from tensors of the checkpoint.
/// </summary>
/// <param name="Layers">The layers with a memory module, ascending.</param>
/// <param name="Order">N-gram length (the token and its N - 1 predecessors).</param>
/// <param name="HeadsPerOrder">Hash heads per n-gram order.</param>
/// <param name="Multipliers">Hash multipliers, N per module.</param>
/// <param name="HeadSizes">Buckets per hash head, (N - 1)·heads_per_order per module (order 2 first).</param>
/// <param name="HeadOffsets">First table row of every hash head, laid out like <paramref name="HeadSizes"/>.</param>
/// <param name="TokenMap">Hashed id of every token id; negative for tokens without one.</param>
/// <param name="Pad">Hashed id that replaces missing tokens.</param>
/// <param name="NormEps">Epsilon of the RMS norms, or null for the model's.</param>
internal sealed record EngramOptions(int[] Layers, int Order, int HeadsPerOrder, long[] Multipliers, long[] HeadSizes, long[] HeadOffsets, int[] TokenMap,
    long Pad, float? NormEps = null)
{
    /// <summary>Hash heads per module.</summary>
    public int Heads => (Order - 1) * HeadsPerOrder;

    /// <summary>The module of layer <paramref name="il"/>, or -1 when it has none.</summary>
    public int ModuleOf(int il) => Array.IndexOf(Layers, il);

    /// <summary>Reads the <c>engram.*</c> keys of <paramref name="source"/>, or null when the model has none.</summary>
    /// <exception cref="InvalidDataException">The tables are inconsistent.</exception>
    public static EngramOptions? Read(ModelSource source)
    {
        int[] layers = source.Arch("engram.layers", Array.Empty<int>());
        if (layers.Length == 0)
        {
            return null;
        }

        int order = source.Arch<int>("engram.order");
        int headsPerOrder = source.Arch<int>("engram.heads_per_order");
        var multipliers = source.Weight("engram.hash_multipliers") is { } multiplierWeight
            ? MemoryMarshal.Cast<byte, long>(multiplierWeight.ReadAll()).ToArray() : source.Arch<long[]>("engram.multipliers");
        var sizes = source.Arch<long[]>("engram.head_sizes");
        var offsets = source.Arch<long[]>("engram.head_offsets");
        var map = source.Weight("engram.token_ids") is { } mapWeight
            ? MemoryMarshal.Cast<byte, int>(mapWeight.ReadAll()).ToArray() : source.Arch<int[]>("engram.token_map");
        long padToken = source.Arch<long>("engram.pad_token");
        int heads = (order - 1) * headsPerOrder;
        if (order < 2 || headsPerOrder < 1 || multipliers.Length != layers.Length * order || sizes.Length != layers.Length * heads
            || offsets.Length != sizes.Length || sizes.Any(s => s < 2) || padToken < 0 || padToken >= map.Length
            || layers.Zip(layers.Skip(1)).Any(p => p.Second <= p.First))
        {
            throw new InvalidDataException($"{source.Architecture}: inconsistent engram tables.");
        }

        float eps = source.Arch("engram.norm_eps", 0f);
        return new EngramOptions(layers, order, headsPerOrder, multipliers, sizes, offsets, map, map[padToken], eps > 0f ? eps : null);
    }
}
