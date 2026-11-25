using Sharplight.Graph;

namespace Sharplight.Shaders.Lowering;

/// <summary>One dispatch of a <see cref="KernelProgram"/>: the kernel variants to choose from, the shared arguments and the buffer ranges it touches.</summary>
public sealed class KernelLaunch
{
    /// <summary>Name for profiling and diagnostics.</summary>
    public required string Name { get; init; }
    /// <summary>Alternative kernels; the first whose condition holds for the bindings runs.</summary>
    public required IReadOnlyList<KernelVariant> Variants { get; init; }
    /// <summary>Arguments shared by all variants, packed in order into the push constants.</summary>
    public required IReadOnlyList<KernelArg> Args { get; init; }
    /// <summary>Ranges read by the launch.</summary>
    public required IReadOnlyList<BufferRange> Reads { get; init; }
    /// <summary>Ranges written by the launch.</summary>
    public required IReadOnlyList<BufferRange> Writes { get; init; }
    /// <summary>
    /// Launches sharing a non-null group write disjoint elements of the ranges they both write (e.g. one output split
    /// between two launches by tile height), so they need no barrier between them for those writes; reads still order
    /// against their writes.
    /// </summary>
    public object? WriteGroup { get; init; }

    /// <summary>Graph node(s) implemented by this launch (for debugging / profiling).</summary>
    public IReadOnlyList<Node> Nodes { get; init; } = [];

    /// <summary>The first variant whose condition holds, or null: the launch is skipped for these bindings.</summary>
    public KernelVariant? Select(Bindings b)
    {
        // Indexed: every new step shape selects the variants of all launches (thousands in a large model's program).
        for (int i = 0; i < Variants.Count; i++)
        {
            if (Variants[i].When(b))
            {
                return Variants[i];
            }
        }

        return null;
    }
}
