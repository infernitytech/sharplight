using Vortice.Vulkan;

namespace Sharplight.Vulkan;

/// <summary>
/// Simple block sub-allocator: large blocks per usage, first-fit with coalescing, dedicated
/// allocations for big resources. Every block is backed by one VkBuffer with device-address usage,
/// so kernels only ever see raw addresses.
/// </summary>
internal sealed unsafe class MemoryAllocator : IDisposable
{
    /// <summary>Weights and state go in big device blocks; the small host-visible staging buffers of inputs/outputs in small ones (a 256 MiB mapped block per usage wastes memory on small models).</summary>
    private static ulong BlockSizeOf(MemoryUsage u) => u == MemoryUsage.Device ? 256UL << 20 : 16UL << 20;
    private const ulong Alignment = 256;
    private readonly VulkanDevice _dev;
    private readonly List<MemoryBlock> _blocks = [];
    private readonly Lock _lock = new();
    private bool _disposed;
    private readonly VkPhysicalDeviceMemoryProperties _props;

    public MemoryAllocator(VulkanDevice dev)
    {
        _dev = dev;
        VkPhysicalDeviceMemoryProperties mp;
        dev.Inst.Api.vkGetPhysicalDeviceMemoryProperties(dev.Phys.Handle, &mp);
        _props = mp;
    }

    public Allocation Allocate(ulong size, MemoryUsage usage)
    {
        size = Math.Max(Alignment, (size + Alignment - 1) & ~(Alignment - 1));
        lock (_lock)
        {
            ulong BlockSize = BlockSizeOf(usage);
            if (size < BlockSize / 2)
            {
                foreach (var b in _blocks)
                {
                    if (b.Usage != usage || b.Dedicated)
                    {
                        continue;
                    }

                    if (TryCarve(b, size, out var a))
                    {
                        return a;
                    }
                }
                var nb = NewBlock(BlockSize, usage, dedicated: false);
                if (TryCarve(nb, size, out var a2))
                {
                    return a2;
                }

                throw new InvalidOperationException("allocator: block carve failed");
            }
            var db = NewBlock(size, usage, dedicated: true);
            db.Free.Clear();
            return new Allocation { Buffer = db.Buffer, Offset = 0, Size = size, Address = db.Address, Mapped = db.Mapped, Block = db, Coherent = db.Coherent };
        }
    }

    public void Free(Allocation a)
    {
        lock (_lock)
        {
            // Disposing the allocator released every block; a model disposed after its device has nothing left to free.
            if (_disposed)
            {
                return;
            }

            var b = a.Block!;
            if (b.Dedicated)
            {
                Destroy(b);
                _blocks.Remove(b);
                return;
            }
            b.Free.Add(a.Offset, a.Size);
            for (int i = 0; i + 1 < b.Free.Count;)
            {
                if (b.Free.Keys[i] + b.Free.Values[i] == b.Free.Keys[i + 1])
                {
                    ulong off = b.Free.Keys[i], sz = b.Free.Values[i] + b.Free.Values[i + 1];
                    b.Free.RemoveAt(i + 1);
                    b.Free[off] = sz;
                }
                else
                {
                    i++;
                }
            }
            // Give a fully free block back to the driver, so unloading a model really returns its memory to the device
            // (other models on the same device, and other applications, can use it).
            if (b.Free.Count == 1 && b.Free.Keys[0] == 0 && b.Free.Values[0] == b.Size)
            {
                Destroy(b);
                _blocks.Remove(b);
            }
        }
    }

    private static bool TryCarve(MemoryBlock b, ulong size, out Allocation a)
    {
        foreach (var (off, sz) in b.Free)
        {
            if (sz < size)
            {
                continue;
            }

            b.Free.Remove(off);
            if (sz > size)
            {
                b.Free.Add(off + size, sz - size);
            }

            a = new Allocation
            {
                Buffer = b.Buffer,
                Offset = off,
                Size = size,
                Address = b.Address + off,
                Mapped = b.Mapped == null ? null : b.Mapped + off,
                Block = b,
                Coherent = b.Coherent,
            };
            return true;
        }
        a = null!;
        return false;
    }

    private MemoryBlock NewBlock(ulong size, MemoryUsage usage, bool dedicated)
    {
        var api = _dev.Api;
        VkBuffer buf;
        // With a transfer queue, buffers are shared by both queue families (outputs are downloaded on the transfer queue).
        fixed (uint* families = _dev.SharingFamilies)
        {
            var bci = new VkBufferCreateInfo
            {
                size = size,
                usage = VkBufferUsageFlags.StorageBuffer | VkBufferUsageFlags.TransferSrc | VkBufferUsageFlags.TransferDst | VkBufferUsageFlags.ShaderDeviceAddress,
                sharingMode = _dev.SharingFamilies.Length > 1 ? VkSharingMode.Concurrent : VkSharingMode.Exclusive,
                queueFamilyIndexCount = (uint)_dev.SharingFamilies.Length,
                pQueueFamilyIndices = _dev.SharingFamilies.Length > 1 ? families : null,
            };
            api.vkCreateBuffer(&bci, null, &buf).CheckResult();
        }

        VkMemoryRequirements req;
        api.vkGetBufferMemoryRequirements(buf, &req);

        var flagsInfo = new VkMemoryAllocateFlagsInfo { flags = VkMemoryAllocateFlags.DeviceAddress };
        var dedicatedInfo = new VkMemoryDedicatedAllocateInfo { buffer = buf, pNext = &flagsInfo };
        VkDeviceMemory mem = default;
        VkMemoryPropertyFlags flags = default;
        var r = VkResult.ErrorOutOfDeviceMemory;
        // Memory types in order of preference: when the preferred heap is full, the next type serves (staging in device-local
        // host-visible memory falls back to plain host memory, e.g. on GPUs without resizable BAR, whose 256 MiB BAR heap is
        // shared by every process on the device).
        foreach (var (typeIndex, typeFlags) in CandidateTypes(req.memoryTypeBits, usage))
        {
            var mai = new VkMemoryAllocateInfo { pNext = dedicated ? &dedicatedInfo : &flagsInfo, allocationSize = req.size, memoryTypeIndex = typeIndex };
            r = api.vkAllocateMemory(&mai, null, &mem);
            if (r == VkResult.Success)
            {
                flags = typeFlags;
                break;
            }
        }
        if (r != VkResult.Success)
        {
            api.vkDestroyBuffer(buf, null);
            throw new OutOfMemoryException($"Vulkan allocation of {size / (1024.0 * 1024):F1} MiB ({usage}) failed: {r}");
        }
        api.vkBindBufferMemory(buf, mem, 0).CheckResult();
        byte* mapped = null;
        if ((flags & VkMemoryPropertyFlags.HostVisible) != 0)
        {
            void* p;
            api.vkMapMemory(mem, 0, VkWholeSize, 0, &p).CheckResult();
            mapped = (byte*)p;
        }
        var bai = new VkBufferDeviceAddressInfo { buffer = buf };
        var block = new MemoryBlock
        {
            Memory = mem,
            Buffer = buf,
            Size = size,
            Address = api.vkGetBufferDeviceAddress(&bai),
            Mapped = mapped,
            Dedicated = dedicated,
            Usage = usage,
            Coherent = (flags & VkMemoryPropertyFlags.HostCoherent) != 0,
            Tracked = (flags & VkMemoryPropertyFlags.DeviceLocal) != 0 ? (long)req.size : 0,
        };
        block.Free.Add(0, size);
        _blocks.Add(block);
        _dev.Track(block.Tracked);
        return block;
    }

    private const ulong VkWholeSize = ulong.MaxValue;

    /// <summary>The memory types for <paramref name="usage"/> among <paramref name="bits"/>, most preferred first, each once.</summary>
    private List<(uint Index, VkMemoryPropertyFlags Flags)> CandidateTypes(uint bits, MemoryUsage usage)
    {
        VkMemoryPropertyFlags[] prefs = usage switch
        {
            MemoryUsage.Device => [VkMemoryPropertyFlags.DeviceLocal],
            MemoryUsage.Upload =>
            [
                VkMemoryPropertyFlags.DeviceLocal | VkMemoryPropertyFlags.HostVisible | VkMemoryPropertyFlags.HostCoherent,
                VkMemoryPropertyFlags.HostVisible | VkMemoryPropertyFlags.HostCoherent,
            ],
            _ =>
            [
                VkMemoryPropertyFlags.HostVisible | VkMemoryPropertyFlags.HostCoherent | VkMemoryPropertyFlags.HostCached,
                VkMemoryPropertyFlags.HostVisible | VkMemoryPropertyFlags.HostCoherent,
            ],
        };
        var types = new List<(uint, VkMemoryPropertyFlags)>();
        foreach (var w in prefs)
        {
            for (uint i = 0; i < _props.memoryTypeCount; i++)
            {
                var f = _props.memoryTypes[(int)i].propertyFlags;
                if ((bits & (1u << (int)i)) == 0 || (f & w) != w || types.Exists(t => t.Item1 == i))
                {
                    continue;
                }

                // For pure device memory avoid host-visible types if a plain one exists.
                if (usage == MemoryUsage.Device && (f & VkMemoryPropertyFlags.HostVisible) != 0 && HasPlainDeviceLocal(bits))
                {
                    continue;
                }

                types.Add((i, f));
            }
        }
        if (types.Count == 0)
        {
            throw new NotSupportedException($"No memory type for {usage}.");
        }

        return types;
    }

    private bool HasPlainDeviceLocal(uint bits)
    {
        for (uint i = 0; i < _props.memoryTypeCount; i++)
        {
            var f = _props.memoryTypes[(int)i].propertyFlags;
            if ((bits & (1u << (int)i)) != 0 && (f & VkMemoryPropertyFlags.DeviceLocal) != 0 && (f & VkMemoryPropertyFlags.HostVisible) == 0)
            {
                return true;
            }
        }
        return false;
    }

    private void Destroy(MemoryBlock b)
    {
        if (b.Mapped != null)
        {
            _dev.Api.vkUnmapMemory(b.Memory);
        }

        _dev.Api.vkDestroyBuffer(b.Buffer, null);
        _dev.Api.vkFreeMemory(b.Memory, null);
        _dev.Track(-b.Tracked);
    }

    public void Dispose()
    {
        lock (_lock)
        {
            _disposed = true;
            foreach (var b in _blocks)
            {
                Destroy(b);
            }

            _blocks.Clear();
        }
    }
}
