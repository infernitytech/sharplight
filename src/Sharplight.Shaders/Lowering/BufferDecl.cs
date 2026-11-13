using Sharplight.Models;

namespace Sharplight.Shaders.Lowering;

/// <summary>A logical buffer of a <see cref="KernelProgram"/>. Backends map it to device memory.</summary>
public sealed class BufferDecl
{
    /// <summary>Index of the buffer in <see cref="KernelProgram.Buffers"/>.</summary>
    public required int Id { get; init; }
    /// <summary>How the buffer is used and which memory it needs.</summary>
    public required BufferKind Kind { get; init; }
    /// <summary>Name for diagnostics (weight, state or value name).</summary>
    public required string Name { get; init; }
    /// <summary>Size in bytes; for activations, the arena high-water mark.</summary>
    public long Bytes { get; set; }
    /// <summary>The model parameter stored in a <see cref="BufferKind.Weight"/> buffer, or null for synthesized constants.</summary>
    public WeightInfo? Weight { get; init; }
    /// <summary>The state held by a <see cref="BufferKind.State"/> buffer.</summary>
    public StateDefinition? State { get; init; }
    /// <summary>Bytes to upload instead of a weight (graph constants), or null.</summary>
    public byte[]? InitialData { get; init; }
    /// <summary>Layout transform applied to the weight bytes on upload (see <see cref="WeightRepacker"/>).</summary>
    public WeightRepack Repack { get; set; }
    /// <summary>Logical elements per row (K) for <see cref="Repack"/>.</summary>
    public long RepackRowElements { get; set; }
    /// <inheritdoc />
    public override string ToString() => $"#{Id} {Kind} {Name} ({Bytes} B{(Repack != WeightRepack.None ? ", " + Repack : "")})";
}
