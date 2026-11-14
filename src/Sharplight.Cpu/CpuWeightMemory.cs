namespace Sharplight.Cpu;

/// <summary>Native memory holding a weight (read from a non-mapped source, or converted), shared between compiled segments.</summary>
internal sealed unsafe class CpuWeightMemory(long bytes)
{
    /// <summary>The memory (64-byte aligned).</summary>
    public byte* Pointer { get; private set; } = Mem.Alloc(bytes, false);

    /// <summary>Size in bytes.</summary>
    public long Bytes => bytes;

    /// <summary>Frees the memory.</summary>
    public void Free()
    {
        Mem.Free(Pointer);
        Pointer = null;
    }
}
