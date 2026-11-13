namespace Sharplight.Cpu;

/// <summary>A memory region (arena slice, state, weight pointer or constant).</summary>
internal sealed unsafe class Buf
{
    public long Bytes;
    public byte* Ptr;
    /// <summary>First/last step touching the buffer (for arena reuse).</summary>
    public int First = int.MaxValue;
    public int Last = -1;
    /// <summary>Lives in the shared activation arena (otherwise Ptr is already final).</summary>
    public bool InArena;
    public long ArenaOffset;
    /// <summary>Keep alive until the end of the segment.</summary>
    public bool Pinned;
    public string Name = "";
}
