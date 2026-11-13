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
    /// <summary>Graph node(s) implemented by this launch (for debugging / profiling).</summary>
    public IReadOnlyList<Node> Nodes { get; init; } = [];

    /// <summary>The first variant whose condition holds, or null: the launch is skipped for these bindings.</summary>
    public KernelVariant? Select(Bindings b) => Variants.FirstOrDefault(v => v.When(b));
}
