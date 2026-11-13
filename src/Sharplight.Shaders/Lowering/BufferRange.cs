namespace Sharplight.Shaders.Lowering;

/// <summary>Byte range of a buffer touched by a launch (used to place barriers).</summary>
/// <param name="Buffer">The buffer.</param>
/// <param name="Start">First byte (inclusive).</param>
/// <param name="End">One past the last byte.</param>
public readonly record struct BufferRange(BufferDecl Buffer, long Start, long End)
{
    /// <summary>True when both ranges cover at least one common byte of the same buffer.</summary>
    public bool Overlaps(BufferRange o) => Buffer == o.Buffer && Start < o.End && o.Start < End;
}
