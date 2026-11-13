using Sharplight.Shaders.Lowering;
using Vortice.Vulkan;

namespace Sharplight.Vulkan;

/// <summary>
/// Streams weight and constant buffers to device memory through two host-visible staging buffers: while the GPU copies one,
/// the CPU fills the other (reading mapped files, repacking rows). Single use, not thread-safe.
/// </summary>
internal sealed unsafe class WeightUploader : IDisposable
{
    private const ulong StagingSize = 128UL << 20;
    private const ulong StagingAlignment = 256;
    private const int Slots = 2;

    private readonly VulkanDevice _dev;
    private readonly VkCommandPool _pool;
    private readonly Allocation[] _staging = new Allocation[Slots];
    private readonly VkCommandBuffer[] _commandBuffers = new VkCommandBuffer[Slots];
    private readonly VkFence[] _fences = new VkFence[Slots];
    private readonly List<(VkBuffer Dst, ulong DstOffset, ulong SrcOffset, ulong Size)> _pending = [];
    private int _slot;
    private ulong _fill;
    private int _created;

    /// <summary>Creates the staging buffers, command buffers (from <paramref name="pool"/>) and fences.</summary>
    internal WeightUploader(VulkanDevice dev, VkCommandPool pool)
    {
        _dev = dev;
        _pool = pool;
        try
        {
            for (int i = 0; i < Slots; i++)
            {
                _staging[i] = dev.Memory.Allocate(StagingSize, MemoryUsage.Upload);
                _created = i + 1;
                var allocateInfo = new VkCommandBufferAllocateInfo { commandPool = pool, level = VkCommandBufferLevel.Primary, commandBufferCount = 1 };
                VkCommandBuffer commandBuffer;
                dev.Api.vkAllocateCommandBuffers(&allocateInfo, &commandBuffer).CheckResult();
                _commandBuffers[i] = commandBuffer;
                var fenceInfo = new VkFenceCreateInfo { flags = VkFenceCreateFlags.Signaled };
                VkFence fence;
                dev.Api.vkCreateFence(&fenceInfo, null, &fence).CheckResult();
                _fences[i] = fence;
            }
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    /// <summary>Copies a whole weight or constant buffer to <paramref name="destination"/>, in chunks that fit the staging buffer.</summary>
    internal void Upload(BufferDecl buffer, Allocation destination)
    {
        long total = buffer.Weight?.ByteSize ?? buffer.InitialData!.Length;
        var source = buffer.Weight?.Source;
        nint pointer = source?.TryGetPointer() ?? 0;
        long done = 0;
        // Repacked weights are transformed in whole rows.
        long rowBytes = buffer.Repack != WeightRepack.None ? WeightRepacker.RowBytes(buffer.Repack, buffer.RepackRowElements) : 1;
        while (done < total)
        {
            if (StagingSize - _fill < (ulong)Math.Min(rowBytes, total - done))
            {
                Flush();
            }

            ulong chunk = Math.Min(StagingSize - _fill, (ulong)(total - done));
            chunk -= chunk % (ulong)rowBytes;
            WeightCopier.CopyChunk(buffer, source, pointer, done, (long)chunk, _staging[_slot].Mapped + _fill, rowBytes);

            _pending.Add((destination.Buffer, destination.Offset + (ulong)done, _fill, chunk));
            _fill = Math.Min(StagingSize, (_fill + chunk + StagingAlignment - 1) & ~(StagingAlignment - 1));
            done += (long)chunk;
        }
    }

    /// <summary>Submits the remaining copies and waits until both staging buffers are idle.</summary>
    internal void Complete()
    {
        Flush();
        for (int i = 0; i < Slots; i++)
        {
            var fence = _fences[i];
            _dev.Api.vkWaitForFences(1, &fence, true, ulong.MaxValue);
        }
    }

    /// <summary>Submits the pending copies from the current staging buffer, then switches to the other one (waiting until it is free).</summary>
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
        _slot ^= 1;
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
    }
}
