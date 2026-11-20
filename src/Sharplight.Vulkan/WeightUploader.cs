using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Shaders.Lowering;
using Vortice.Vulkan;

namespace Sharplight.Vulkan;

/// <summary>
/// Streams weight and constant buffers to device memory. Destinations the host can map (integrated GPUs and other unified-memory
/// devices) are written directly; the others go through up to two host-visible staging buffers: while the GPU copies one, the CPU
/// fills the other (reading mapped files, repacking rows). Not thread-safe; reusable after <see cref="Complete"/> (the streaming
/// uploader of <see cref="VulkanDevice.StreamUploader"/>).
/// </summary>
/// <remarks>
/// The staging buffers are sized to what is free, not fixed: at most <see cref="MaxStagingSize"/>, no more than the uploads need, at
/// most half the free budget of the preferred staging heap, and halved on every failed allocation down to
/// <see cref="MinStagingSize"/> (one buffer when a second does not fit), so a small or crowded BAR window, or a small device, still
/// loads, just with more submissions.
/// </remarks>
internal sealed unsafe class WeightUploader : IDisposable
{
    private const ulong MaxStagingSize = 128UL << 20;
    private const ulong MinStagingSize = 1UL << 20;
    private const ulong StagingAlignment = 256;
    /// <summary>Bytes written per call when copying straight into mapped device memory (the copy routines take int-sized spans).</summary>
    private const long DirectChunk = 64L << 20;
    private const int MaxSlots = 2;

    private readonly VulkanDevice _dev;
    private readonly VkCommandPool _pool;
    private readonly long _totalBytes;
    private readonly long _largestRow;
    private readonly Allocation[] _staging = new Allocation[MaxSlots];
    private readonly VkCommandBuffer[] _commandBuffers = new VkCommandBuffer[MaxSlots];
    private readonly VkFence[] _fences = new VkFence[MaxSlots];
    private readonly List<(VkBuffer Dst, ulong DstOffset, ulong SrcOffset, ulong Size)> _pending = [];
    private ulong _stagingSize;
    private int _slot;
    private ulong _fill;
    private int _created;

    /// <summary>Prepares an upload of <paramref name="totalBytes"/> bytes; staging memory is allocated on the first copy that needs it.</summary>
    /// <param name="dev">The device.</param>
    /// <param name="pool">The command pool of the copies.</param>
    /// <param name="totalBytes">Bytes of all buffers to upload (bounds the staging size).</param>
    /// <param name="largestRow">The largest repacked row, which must fit one staging buffer.</param>
    internal WeightUploader(VulkanDevice dev, VkCommandPool pool, long totalBytes, long largestRow)
    {
        _dev = dev;
        _pool = pool;
        _totalBytes = totalBytes;
        _largestRow = Math.Max(1, largestRow);
    }

    /// <summary>Bytes per staging buffer (0 until a staged copy allocated them).</summary>
    internal ulong StagingSize => _stagingSize;

    /// <summary>Number of staging buffers (0 until a staged copy allocated them).</summary>
    internal int StagingBuffers => _created;

    /// <summary>Copies a whole weight or constant buffer to <paramref name="destination"/>, in chunks that fit the staging buffer.</summary>
    internal void Upload(BufferDecl buffer, Allocation destination)
    {
        long total = buffer.Weight?.ByteSize ?? buffer.InitialData!.Length;
        var source = buffer.Weight?.Source;
        nint pointer = source?.TryGetPointer() ?? 0;
        // Repacked weights are transformed in whole rows.
        long rowBytes = buffer.Repack != WeightRepack.None ? WeightRepacker.RowBytes(buffer.Repack, buffer.RepackRowElements) : 1;
        if (destination.Mapped != null)
        {
            UploadDirect(buffer, source, pointer, total, rowBytes, destination.Mapped);
            return;
        }

        EnsureStaging();
        long done = 0;
        while (done < total)
        {
            if (_stagingSize - _fill < (ulong)Math.Min(rowBytes, total - done))
            {
                Flush();
            }

            ulong chunk = Math.Min(_stagingSize - _fill, (ulong)(total - done));
            chunk -= chunk % (ulong)rowBytes;
            if (buffer.Repack != WeightRepack.None)
            {
                // Repacking writes out of order, which write-combined staging memory makes slow: repack into cached memory, then one
                // sequential copy.
                byte* repacked = RepackScratch(chunk);
                WeightCopier.CopyChunk(buffer, source, pointer, done, (long)chunk, repacked, rowBytes);
                WeightCopier.CopyParallel(repacked, _staging[_slot].Mapped + _fill, (long)chunk);
            }
            else
            {
                WeightCopier.CopyChunk(buffer, source, pointer, done, (long)chunk, _staging[_slot].Mapped + _fill, rowBytes);
            }

            _pending.Add((destination.Buffer, destination.Offset + (ulong)done, _fill, chunk));
            _fill = Math.Min(_stagingSize, (_fill + chunk + StagingAlignment - 1) & ~(StagingAlignment - 1));
            done += (long)chunk;
        }
    }

    private byte* _repackScratch;
    private ulong _repackScratchBytes;

    /// <summary>Cached host memory of at least <paramref name="bytes"/> for repacking a chunk before it goes to staging memory, kept until disposed.</summary>
    private byte* RepackScratch(ulong bytes)
    {
        if (_repackScratchBytes < bytes)
        {
            System.Runtime.InteropServices.NativeMemory.AlignedFree(_repackScratch);
            _repackScratch = (byte*)System.Runtime.InteropServices.NativeMemory.AlignedAlloc((nuint)bytes, 4096);
            _repackScratchBytes = bytes;
        }

        return _repackScratch;
    }

    /// <summary>Submits the remaining copies and waits until the staging buffers are idle.</summary>
    internal void Complete()
    {
        Flush();
        for (int i = 0; i < _created; i++)
        {
            var fence = _fences[i];
            _dev.Api.vkWaitForFences(1, &fence, true, ulong.MaxValue);
        }
    }

    /// <summary>Writes a buffer straight into its host-visible destination (unified memory): no staging memory, no copy on the device.</summary>
    private static void UploadDirect(BufferDecl buffer, IWeightSource? source, nint pointer, long total, long rowBytes, byte* destination)
    {
        long step = Math.Max(rowBytes, DirectChunk - DirectChunk % rowBytes);
        for (long done = 0; done < total; done += step)
        {
            long chunk = Math.Min(step, total - done);
            WeightCopier.CopyChunk(buffer, source, pointer, done, chunk, destination + done, rowBytes);
        }
    }

    /// <summary>Allocates the staging buffers, as large as fits (see the remarks of the type).</summary>
    private void EnsureStaging()
    {
        if (_created > 0)
        {
            return;
        }

        ulong needed = Math.Max((ulong)_totalBytes, (ulong)_largestRow);
        ulong minimum = Math.Max(MinStagingSize, RoundUp((ulong)_largestRow));
        ulong size = Math.Min(MaxStagingSize, Math.Min(RoundUp(needed), _dev.Memory.MaxAllocation));
        size = Math.Min(size, Math.Max(minimum, _dev.Memory.StagingHeadroom() / MaxSlots));
        size = Math.Max(size, minimum);
        DeviceOutOfMemoryException? failure = null;
        while (_created == 0)
        {
            try
            {
                _staging[0] = _dev.Memory.Allocate(size, MemoryUsage.Upload, "weight upload staging");
                CreateSlot(0);
            }
            catch (DeviceOutOfMemoryException error) when (size / 2 >= minimum)
            {
                failure = error;
                size /= 2;
            }
        }

        _stagingSize = size;
        // Double buffering only pays when there is more than one buffer's worth to upload, and only when it fits.
        if ((ulong)_totalBytes > size)
        {
            try
            {
                _staging[1] = _dev.Memory.Allocate(size, MemoryUsage.Upload, "weight upload staging");
                CreateSlot(1);
            }
            catch (DeviceOutOfMemoryException)
            {
                // One buffer: the CPU waits for each copy before refilling it.
            }
        }

        if (failure is not null && DiagnosticSwitches.LoadTiming)
        {
            Console.Error.WriteLine($"[load] upload staging reduced to {_created} x {size >> 20} MiB: {failure.Message}");
        }
    }

    private void CreateSlot(int i)
    {
        try
        {
            var allocateInfo = new VkCommandBufferAllocateInfo { commandPool = _pool, level = VkCommandBufferLevel.Primary, commandBufferCount = 1 };
            VkCommandBuffer commandBuffer;
            _dev.Api.vkAllocateCommandBuffers(&allocateInfo, &commandBuffer).CheckResult();
            _commandBuffers[i] = commandBuffer;
            var fenceInfo = new VkFenceCreateInfo { flags = VkFenceCreateFlags.Signaled };
            VkFence fence;
            _dev.Api.vkCreateFence(&fenceInfo, null, &fence).CheckResult();
            _fences[i] = fence;
        }
        finally
        {
            _created = i + 1;
        }
    }

    private static ulong RoundUp(ulong bytes) => (bytes + StagingAlignment - 1) & ~(StagingAlignment - 1);

    /// <summary>Submits the pending copies from the current staging buffer, then switches to the next one (waiting until it is free).</summary>
    private void Flush()
    {
        if (_pending.Count == 0)
        {
            return;
        }

        var commandBuffer = _commandBuffers[_slot];
        var beginInfo = new VkCommandBufferBeginInfo { flags = VkCommandBufferUsageFlags.OneTimeSubmit };
        _dev.Api.vkBeginCommandBuffer(commandBuffer, &beginInfo).CheckResult();
        foreach (var (destination, destinationOffset, sourceOffset, size) in _pending)
        {
            var region = new VkBufferCopy { srcOffset = _staging[_slot].Offset + sourceOffset, dstOffset = destinationOffset, size = size };
            _dev.Api.vkCmdCopyBuffer(commandBuffer, _staging[_slot].Buffer, destination, 1, &region);
        }

        _dev.Api.vkEndCommandBuffer(commandBuffer).CheckResult();
        var fence = _fences[_slot];
        _dev.Api.vkResetFences(1, &fence).CheckResult();
        _dev.Submit(commandBuffer, fence);
        _pending.Clear();
        _slot = (_slot + 1) % _created;
        _fill = 0;
        var next = _fences[_slot];
        _dev.Api.vkWaitForFences(1, &next, true, ulong.MaxValue).CheckResult();
    }

    /// <summary>Frees the staging memory, command buffers and fences. Callers must have completed or abandoned the uploads.</summary>
    public void Dispose()
    {
        for (int i = 0; i < _created; i++)
        {
            if (_fences[i].Handle != 0)
            {
                _dev.Api.vkDestroyFence(_fences[i], null);
            }

            if (_commandBuffers[i].Handle != 0)
            {
                var commandBuffer = _commandBuffers[i];
                _dev.Api.vkFreeCommandBuffers(_pool, 1, &commandBuffer);
            }

            _dev.Memory.Free(_staging[i]);
        }

        _created = 0;
        System.Runtime.InteropServices.NativeMemory.AlignedFree(_repackScratch);
        _repackScratch = null;
        _repackScratchBytes = 0;
    }
}
