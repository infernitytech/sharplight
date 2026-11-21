using Vortice.Vulkan;

namespace Sharplight.Vulkan;

/// <summary>
/// Host memory the device imports (VK_EXT_external_memory_host), so that copies read it directly, without staging on the host: the mapped
/// pages of streamed weights, from which the streamed layers and the selected experts of split passes are copied at every execution. An
/// import pins its pages like page-locked memory, so the imports stay within the pinned-memory limit (else a quarter of the memory
/// available to the process). Ranges are imported once and shared by their holders. Thread-safe.
/// </summary>
internal sealed unsafe class HostMemoryImports : IDisposable
{
    /// <summary>Largest single import: drivers limit allocations and buffers (maxMemoryAllocationSize, maxBufferSize).</summary>
    private const long MaxChunk = 1L << 30;

    private readonly VulkanDevice _dev;
    private readonly Dictionary<nint, Entry> _entries = [];
    private long _importedBytes;

    internal HostMemoryImports(VulkanDevice dev)
    {
        _dev = dev;
        Alignment = dev.Phys.HostPointerAlignment;
    }

    /// <summary>The alignment of imported addresses and sizes.</summary>
    internal long Alignment { get; }

    private long Budget => _dev.Limits.PinnedHostMemory ?? GC.GetGCMemoryInfo().TotalAvailableMemoryBytes / 4;

    /// <summary>
    /// Imports the aligned part of [<paramref name="start"/>, + <paramref name="bytes"/>) (the whole aligned units inside it), or gains a hold
    /// on the import made before for the same range. False when nothing was imported: the range holds no aligned unit, the budget is used up
    /// or the driver refuses (copies then go through staging).
    /// </summary>
    internal bool TryImport(byte* start, long bytes)
    {
        lock (_entries)
        {
            if (_entries.TryGetValue((nint)start, out var existing))
            {
                existing.Holders++;
                return true;
            }

            long first = ((long)start + Alignment - 1) & ~(Alignment - 1);
            long end = ((long)start + bytes) & ~(Alignment - 1);
            if (end - first < Alignment || _importedBytes + (end - first) > Budget)
            {
                return false;
            }

            var entry = new Entry();
            for (long chunk = first; chunk < end; chunk += MaxChunk)
            {
                long size = Math.Min(MaxChunk, end - chunk);
                if (!TryImportChunk((byte*)chunk, size, out var buffer, out var memory))
                {
                    Destroy(entry);
                    return false;
                }

                entry.Chunks.Add(new ImportedChunk((nint)chunk, size, buffer, memory));
            }

            _entries[(nint)start] = entry;
            _importedBytes += end - first;
            return true;
        }
    }

    /// <summary>Gives up a hold from <see cref="TryImport"/>; the last one frees the import.</summary>
    internal void Release(byte* start)
    {
        lock (_entries)
        {
            if (!_entries.TryGetValue((nint)start, out var entry) || --entry.Holders > 0)
            {
                return;
            }

            _entries.Remove((nint)start);
            _importedBytes -= entry.Chunks.Sum(c => c.Bytes);
            Destroy(entry);
        }
    }

    /// <summary>
    /// Splits [<paramref name="start"/>, + <paramref name="bytes"/>) into pieces in order: those inside an import with their buffer and offset,
    /// and the gaps between them (<see cref="HostPiece.Buffer"/> null) that must be staged.
    /// </summary>
    internal void Split(byte* start, long bytes, List<HostPiece> pieces)
    {
        pieces.Clear();
        long position = (long)start;
        long end = position + bytes;
        lock (_entries)
        {
            while (position < end)
            {
                var chunk = Find(position);
                if (chunk is { } inside)
                {
                    long pieceEnd = Math.Min(end, (long)inside.Start + inside.Bytes);
                    pieces.Add(new HostPiece((nint)position, pieceEnd - position, inside.Buffer, (ulong)(position - (long)inside.Start)));
                    position = pieceEnd;
                    continue;
                }

                long next = NextStart(position, end);
                pieces.Add(new HostPiece((nint)position, next - position, VkBuffer.Null, 0));
                position = next;
            }
        }
    }

    /// <summary>The imported chunk that holds <paramref name="address"/>, if any.</summary>
    private ImportedChunk? Find(long address)
    {
        foreach (var entry in _entries.Values)
        {
            foreach (var chunk in entry.Chunks)
            {
                if (address >= (long)chunk.Start && address < (long)chunk.Start + chunk.Bytes)
                {
                    return chunk;
                }
            }
        }

        return null;
    }

    /// <summary>The start of the first imported chunk in (<paramref name="address"/>, <paramref name="end"/>), else <paramref name="end"/>.</summary>
    private long NextStart(long address, long end)
    {
        long next = end;
        foreach (var entry in _entries.Values)
        {
            foreach (var chunk in entry.Chunks)
            {
                if ((long)chunk.Start > address && (long)chunk.Start < next)
                {
                    next = (long)chunk.Start;
                }
            }
        }

        return next;
    }

    private bool TryImportChunk(byte* start, long bytes, out VkBuffer buffer, out VkDeviceMemory memory)
    {
        buffer = VkBuffer.Null;
        memory = VkDeviceMemory.Null;
        var api = _dev.Api;
        var pointerProperties = new VkMemoryHostPointerPropertiesEXT();
        if (api.vkGetMemoryHostPointerPropertiesEXT(VkExternalMemoryHandleTypeFlags.HostAllocationEXT, start, &pointerProperties) != VkResult.Success)
        {
            return false;
        }

        var external = new VkExternalMemoryBufferCreateInfo { handleTypes = VkExternalMemoryHandleTypeFlags.HostAllocationEXT };
        var bufferInfo = new VkBufferCreateInfo { pNext = &external, size = (ulong)bytes, usage = VkBufferUsageFlags.TransferSrc, sharingMode = VkSharingMode.Exclusive };
        VkBuffer created;
        if (api.vkCreateBuffer(&bufferInfo, null, &created) != VkResult.Success)
        {
            return false;
        }

        VkMemoryRequirements requirements;
        api.vkGetBufferMemoryRequirements(created, &requirements);
        uint types = requirements.memoryTypeBits & pointerProperties.memoryTypeBits;
        if (types == 0)
        {
            api.vkDestroyBuffer(created, null);
            return false;
        }

        var import = new VkImportMemoryHostPointerInfoEXT { handleType = VkExternalMemoryHandleTypeFlags.HostAllocationEXT, pHostPointer = start };
        var allocateInfo = new VkMemoryAllocateInfo { pNext = &import, allocationSize = (ulong)bytes, memoryTypeIndex = (uint)System.Numerics.BitOperations.TrailingZeroCount(types) };
        VkDeviceMemory allocated;
        if (api.vkAllocateMemory(&allocateInfo, null, &allocated) != VkResult.Success)
        {
            api.vkDestroyBuffer(created, null);
            return false;
        }

        if (api.vkBindBufferMemory(created, allocated, 0) != VkResult.Success)
        {
            api.vkFreeMemory(allocated, null);
            api.vkDestroyBuffer(created, null);
            return false;
        }

        buffer = created;
        memory = allocated;
        return true;
    }

    private void Destroy(Entry entry)
    {
        foreach (var chunk in entry.Chunks)
        {
            _dev.Api.vkDestroyBuffer(chunk.Buffer, null);
            _dev.Api.vkFreeMemory(chunk.Memory, null);
        }

        entry.Chunks.Clear();
    }

    /// <summary>Frees every import (the device must be idle).</summary>
    public void Dispose()
    {
        lock (_entries)
        {
            foreach (var entry in _entries.Values)
            {
                Destroy(entry);
            }

            _entries.Clear();
            _importedBytes = 0;
        }
    }

    /// <summary>One imported range: its chunks and holders.</summary>
    private sealed class Entry
    {
        public List<ImportedChunk> Chunks { get; } = [];
        public int Holders { get; set; } = 1;
    }

    /// <summary>One import of at most <see cref="MaxChunk"/> bytes.</summary>
    private readonly record struct ImportedChunk(nint Start, long Bytes, VkBuffer Buffer, VkDeviceMemory Memory);
}
