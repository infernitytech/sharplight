using Sharplight.Models;

namespace Sharplight.Shaders.Lowering;

/// <summary>A logical buffer of a <see cref="KernelProgram"/>. Backends map it to device memory.</summary>
public sealed class BufferDecl
{
    public required int Id { get; init; }
    public required BufferKind Kind { get; init; }
    public required string Name { get; init; }
    public long Bytes { get; set; }
    public WeightInfo? Weight { get; init; }
    public StateDefinition? State { get; init; }
    public byte[]? InitialData { get; init; }
    /// <summary>Layout transform applied to the weight bytes on upload (see <see cref="WeightRepacker"/>).</summary>
    public WeightRepack Repack { get; set; }
    /// <summary>Logical elements per row (K) for <see cref="Repack"/>.</summary>
    public long RepackRowElements { get; set; }
    public override string ToString() => $"#{Id} {Kind} {Name} ({Bytes} B{(Repack != WeightRepack.None ? ", " + Repack : "")})";
}
