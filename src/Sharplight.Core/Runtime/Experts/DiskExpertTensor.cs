namespace Sharplight.Runtime.Experts;

/// <summary>
/// A routed-expert weight [experts, N, K] left in its file (<see cref="ExpertStore"/>): expert <c>e</c> is the contiguous range of
/// <see cref="ExpertBytes"/> bytes at <see cref="FileOffset"/> + e * <see cref="ExpertBytes"/>, read into a cache slot when a pass selects it.
/// </summary>
internal sealed class DiskExpertTensor
{
    public DiskExpertTensor(string name, string group, IExpertFile file, long fileOffset, long expertBytes, int experts, int slotsPerRow)
    {
        Name = name;
        Group = group;
        File = file;
        FileOffset = fileOffset;
        ExpertBytes = expertBytes;
        Experts = experts;
        SlotsPerRow = slotsPerRow;
        int alignment = Math.Max(1, file.Alignment);
        // The aligned span of any expert: its bytes rounded up, plus one alignment unit when it starts inside one.
        SlotBytes = (expertBytes + alignment - 1) / alignment * alignment + (alignment > 1 ? alignment : 0);
        SlotBytes = (SlotBytes + ExpertFileHandle.DirectAlignment - 1) / ExpertFileHandle.DirectAlignment * ExpertFileHandle.DirectAlignment;
        SlotOf = new int[experts];
        Array.Fill(SlotOf, -1);
    }

    /// <summary>The weight's name (for logs).</summary>
    public string Name { get; }

    /// <summary>The placement group of the layer's routed experts (<c>&lt;layer&gt;#experts</c>): the tensors read together.</summary>
    public string Group { get; }

    /// <summary>The open file.</summary>
    public IExpertFile File { get; }

    /// <summary>Offset of expert 0 in the file.</summary>
    public long FileOffset { get; }

    /// <summary>Bytes of one expert.</summary>
    public long ExpertBytes { get; }

    /// <summary>Experts in the weight.</summary>
    public int Experts { get; }

    /// <summary>Experts a row selects (the slots of the layer's router).</summary>
    public int SlotsPerRow { get; }

    /// <summary>Bytes of a cache slot that holds any one expert with the reads aligned for the file.</summary>
    public long SlotBytes { get; }

    /// <summary>The pool the tensor's experts are cached in (set by the store's layout).</summary>
    public ExpertSlotPool? Pool { get; set; }

    /// <summary>Per expert: its slot in <see cref="Pool"/>, or -1.</summary>
    public int[] SlotOf { get; }

    /// <summary>Where expert <paramref name="expert"/>'s aligned read starts in the file and where its bytes start within that read.</summary>
    public (long ReadOffset, long Skip, long ReadBytes) ReadRange(int expert)
    {
        long start = FileOffset + expert * ExpertBytes;
        long alignment = Math.Max(1, File.Alignment);
        long readOffset = start / alignment * alignment;
        long end = (start + ExpertBytes + alignment - 1) / alignment * alignment;
        return (readOffset, start - readOffset, end - readOffset);
    }
}
