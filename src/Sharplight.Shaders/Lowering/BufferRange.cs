namespace Sharplight.Shaders.Lowering;

/// <summary>Byte range of a buffer touched by a launch (used to place barriers).</summary>
public readonly record struct BufferRange(BufferDecl Buffer, long Start, long End)
{
    public bool Overlaps(BufferRange o) => Buffer == o.Buffer && Start < o.End && o.Start < End;
}
