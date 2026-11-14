using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// Settings of <see cref="HierarchicalCycles"/>, read from the <c>{arch}.cycles.*</c> keys: <c>stack_layers</c> (S, 0 = off), <c>high</c> (H) and
/// <c>low</c> (L). The physical layers are two stacks of S layers each, the low stack (layers [0, S)) and the high stack (layers [S, 2S)).
/// </summary>
/// <param name="StackLayers">Layers per stack S.</param>
/// <param name="HighCycles">High cycles H: each runs the low stack L times, then the high stack once.</param>
/// <param name="LowCycles">Low passes L per high cycle.</param>
internal sealed record HierarchicalCycleOptions(int StackLayers, int HighCycles, int LowCycles)
{
    /// <summary>Reads the options of <paramref name="src"/>; null when the model has no hierarchical cycles.</summary>
    /// <exception cref="InvalidDataException">The cycle counts are not positive or the layer count is not two stacks.</exception>
    public static HierarchicalCycleOptions? Read(ModelSource src, int layerCount)
    {
        int stackLayers = src.Arch("cycles.stack_layers", 0);
        if (stackLayers <= 0)
        {
            return null;
        }

        var options = new HierarchicalCycleOptions(stackLayers, src.Arch("cycles.high", 1), src.Arch("cycles.low", 1));
        if (options.HighCycles < 1 || options.LowCycles < 1 || layerCount != 2 * stackLayers)
        {
            throw new InvalidDataException("decoder: cycles.high and cycles.low must be positive and layer_count must be 2 · cycles.stack_layers.");
        }

        return options;
    }

    /// <summary>Stack passes (each with KV caches of its own): H · (L + 1) passes of S layers.</summary>
    public int Slots => StackLayers * HighCycles * (LowCycles + 1);

    /// <summary>
    /// The physical layer that cache slot <paramref name="slot"/> runs. Slots are numbered pass by pass: pass (h, l) for l &lt; L is a low-stack pass,
    /// pass (h, L) the high-stack pass of cycle h, and slot = (h · (L + 1) + l) · S + layer.
    /// </summary>
    public int PhysicalLayerOf(int slot)
    {
        int pass = slot / StackLayers;
        bool high = pass % (LowCycles + 1) == LowCycles;
        return (high ? StackLayers : 0) + slot % StackLayers;
    }
}
