using Sharplight.Models;

namespace Sharplight.Runtime.Experts;

/// <summary>
/// A routed-expert weight [experts, N, K] left in its file (<see cref="ExpertStore"/>): expert <c>e</c> is the contiguous range of
/// <see cref="StoredExpertBytes"/> bytes at <see cref="FileOffset"/> + e * <see cref="StoredExpertBytes"/>, read into a cache slot when a pass
/// selects it. With an <see cref="Encoding"/> the stored blocks differ from the weight's data type: the slot then holds the read and, behind it
/// at <see cref="DecodedOffset"/>, the <see cref="ExpertBytes"/> it decodes to.
/// </summary>
internal sealed class DiskExpertTensor
{
    public DiskExpertTensor(string name, string group, IExpertFile file, long fileOffset, long expertBytes, int experts, int slotsPerRow,
        IStoredBlockEncoding? encoding = null)
    {
        Name = name;
        Group = group;
        File = file;
        FileOffset = fileOffset;
        ExpertBytes = expertBytes;
        Encoding = encoding;
        StoredExpertBytes = encoding is null ? expertBytes : expertBytes / encoding.BlockBytes * encoding.StoredBlockBytes;
        Experts = experts;
        SlotsPerRow = slotsPerRow;
        int alignment = Math.Max(1, file.Alignment);
        // The aligned span of any expert: its bytes rounded up, plus one alignment unit when it starts inside one.
        long readSpan = (StoredExpertBytes + alignment - 1) / alignment * alignment + (alignment > 1 ? alignment : 0);
        readSpan = RoundUp(readSpan, ExpertFileHandle.DirectAlignment);
        DecodedOffset = encoding is null ? -1 : readSpan;
        SlotBytes = encoding is null ? readSpan : RoundUp(readSpan + expertBytes, ExpertFileHandle.DirectAlignment);
        SlotOf = new int[experts];
        Array.Fill(SlotOf, -1);
    }

    private static long RoundUp(long value, long unit) => (value + unit - 1) / unit * unit;

    /// <summary>The weight's name (for logs).</summary>
    public string Name { get; }

    /// <summary>The placement group of the layer's routed experts (<c>&lt;layer&gt;#experts</c>): the tensors read together.</summary>
    public string Group { get; }

    /// <summary>The open file.</summary>
    public IExpertFile File { get; }

    /// <summary>Offset of expert 0 in the file.</summary>
    public long FileOffset { get; }

    /// <summary>Bytes of one expert in the weight's data type (what a pass reads from the slot).</summary>
    public long ExpertBytes { get; }

    /// <summary>Bytes of one expert in the file (<see cref="ExpertBytes"/> unless the weight has an <see cref="Encoding"/>).</summary>
    public long StoredExpertBytes { get; }

    /// <summary>How the stored blocks decode to the weight's data type, or null when they are stored as they are used.</summary>
    public IStoredBlockEncoding? Encoding { get; }

    /// <summary>With an <see cref="Encoding"/>: where in a slot the decoded expert starts; -1 without one.</summary>
    public long DecodedOffset { get; }

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
        long start = FileOffset + expert * StoredExpertBytes;
        long alignment = Math.Max(1, File.Alignment);
        long readOffset = start / alignment * alignment;
        long end = (start + StoredExpertBytes + alignment - 1) / alignment * alignment;
        return (readOffset, start - readOffset, end - readOffset);
    }

    /// <summary>Where expert <paramref name="expert"/>'s bytes in the weight's data type start within its slot.</summary>
    public long DataOffset(int expert) => Encoding is null ? ReadRange(expert).Skip : DecodedOffset;

    /// <summary>Decodes the expert read into <paramref name="slot"/> behind the read (<see cref="DecodedOffset"/>); nothing without an encoding.</summary>
    public unsafe void Decode(int expert, byte* slot)
    {
        if (Encoding is null)
        {
            return;
        }

        Encoding.Decode(new ReadOnlySpan<byte>(slot + ReadRange(expert).Skip, checked((int)StoredExpertBytes)),
            new Span<byte>(slot + DecodedOffset, checked((int)ExpertBytes)));
    }
}
