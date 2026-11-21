using Sharplight.Graph;

namespace Sharplight.Llm;

/// <summary>
/// Window-sized KV caches for a single-sequence session (<see cref="Runtime.DeviceFeatures.WindowKv"/>): the session's graph is the
/// one-slot paged form of the decode graph (<see cref="MultiSequence.Transform(ComputeGraph, int, int, int, int, int, int)"/>), so the
/// caches only sliding-window attention reads (<see cref="MultiSequence.WindowedCaches"/>) keep a ring of <see cref="WindowRows"/>
/// rows instead of a row per position. Both block tables are fixed: the full caches map position p to row p (identity), the window
/// caches map block b to block b mod (<see cref="WindowRows"/> / <see cref="BlockSize"/>), so a window row is reused once every query
/// that can see it has run. The ring holds every position one pass can see (the window before its first row and its own rows), so a
/// session that only appends (it has no truncation) computes exactly what full-length caches compute; the paged kernels are the ones
/// the serving decoder runs.
/// </summary>
internal sealed class SessionKvLayout
{
    /// <summary>Rows per block of both tables.</summary>
    public int BlockSize { get; }

    /// <summary>Columns of each block table (the context in blocks).</summary>
    public int BlocksPerSlot { get; }

    /// <summary>Keys a window layer's query sees.</summary>
    public int Window { get; }

    /// <summary>Rows of each window-sized cache (whole blocks); 0 when every cache has a row per position.</summary>
    public int WindowRows { get; }

    /// <summary>Rows of each full cache (the context rounded up to whole blocks).</summary>
    public long CacheRows => (long)BlocksPerSlot * BlockSize;

    private SessionKvLayout(int blockSize, int blocksPerSlot, int window, int windowRows)
    {
        BlockSize = blockSize;
        BlocksPerSlot = blocksPerSlot;
        Window = window;
        WindowRows = windowRows;
    }

    /// <summary>
    /// The layout of a session over <paramref name="graph"/> with a context of <paramref name="contextLength"/> rows and passes of up to
    /// <paramref name="maxTokens"/> rows, or null when the graph has no window-sized caches, a ring would not be smaller than the context,
    /// or the graph has no multi-sequence form.
    /// </summary>
    public static SessionKvLayout? For(ComputeGraph graph, int contextLength, int maxTokens, int blockSize)
    {
        if (blockSize <= 0 || contextLength <= 0 || !MultiSequence.IsSupported(graph, out _))
        {
            return null;
        }

        var (window, caches) = MultiSequence.WindowedCaches(graph);
        if (caches.Count == 0)
        {
            return null;
        }

        int blocksPerSlot = (contextLength + blockSize - 1) / blockSize;
        int ringBlocks = RingBlocks(window, Math.Min(maxTokens, contextLength), blockSize);
        return ringBlocks < blocksPerSlot ? new SessionKvLayout(blockSize, blocksPerSlot, window, ringBlocks * blockSize) : null;
    }

    /// <summary>
    /// The one-slot paged form of a session over a context of <paramref name="contextLength"/> rows with a row per position in every cache
    /// (no ring): the reference the window-sized layout is checked against (same kernels, so the same results bit for bit).
    /// </summary>
    internal static SessionKvLayout FullLength(int contextLength, int blockSize) => new(blockSize, (contextLength + blockSize - 1) / blockSize, 0, 0);

    /// <summary>
    /// Blocks of a ring that holds every position one pass of <paramref name="rows"/> rows reads or writes: a pass starting at position p
    /// touches positions [p − window + 1, p + rows − 1], window + rows − 1 positions over at most one partial block at each end.
    /// </summary>
    internal static int RingBlocks(int window, int rows, int blockSize) => (window + rows - 1 + blockSize - 1) / blockSize + 1;

    /// <summary>The full caches' block table [1, <see cref="BlocksPerSlot"/>]: block b is row block b.</summary>
    public int[] BlockTable() => [.. Enumerable.Range(0, BlocksPerSlot)];

    /// <summary>The window caches' block table [1, <see cref="BlocksPerSlot"/>]: block b is ring block b mod (window rows / block size).</summary>
    public int[] WindowBlockTable()
    {
        int ring = WindowRows / BlockSize;
        return [.. Enumerable.Range(0, BlocksPerSlot).Select(b => b % ring)];
    }

    /// <summary>Transforms <paramref name="graph"/> into the session's one-slot paged form with window-sized caches.</summary>
    public MultiSequence.Result Transform(ComputeGraph graph) => MultiSequence.Transform(graph, 1, BlockSize, BlocksPerSlot, 0, 0, WindowRows);
}
