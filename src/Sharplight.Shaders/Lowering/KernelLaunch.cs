using Sharplight.Graph;

namespace Sharplight.Shaders.Lowering;

public sealed class KernelLaunch
{
    public required string Name { get; init; }
    public required IReadOnlyList<KernelVariant> Variants { get; init; }
    public required IReadOnlyList<KernelArg> Args { get; init; }
    public required IReadOnlyList<BufferRange> Reads { get; init; }
    public required IReadOnlyList<BufferRange> Writes { get; init; }
    /// <summary>Graph node(s) implemented by this launch (for debugging / profiling).</summary>
    public IReadOnlyList<Node> Nodes { get; init; } = [];

    /// <summary>The first variant whose condition holds, or null: the launch is skipped for these bindings.</summary>
    public KernelVariant? Select(Bindings b) => Variants.FirstOrDefault(v => v.When(b));
}
