using Vortice.Vulkan;

namespace Sharplight.Vulkan;

internal enum MemoryUsage
{
    /// <summary>Device local, GPU only.</summary>
    Device,
    /// <summary>Host writes, device reads (prefers device-local host-visible "ReBAR" memory).</summary>
    Upload,
    /// <summary>Device writes, host reads (prefers host-cached memory).</summary>
    Readback,
}

/// <summary>A sub-allocation addressed by device address.</summary>
internal sealed unsafe class Allocation
{
    public required VkBuffer Buffer { get; init; }
    public required ulong Offset { get; init; }
    public required ulong Size { get; init; }
    public required ulong Address { get; init; }
    /// <summary>Host pointer for host-visible memory, else null.</summary>
    public byte* Mapped { get; init; }
    internal MemoryBlock? Block { get; init; }
    internal bool Coherent { get; init; } = true;
}

internal sealed unsafe class MemoryBlock
{
    public required VkDeviceMemory Memory { get; init; }
    public required VkBuffer Buffer { get; init; }
    public required ulong Size { get; init; }
    public required ulong Address { get; init; }
    public byte* Mapped { get; init; }
    public bool Dedicated { get; init; }
    public bool Coherent { get; init; }
    public MemoryUsage Usage { get; init; }
    public readonly SortedList<ulong, ulong> Free = [];   // offset → size
}

/// <summary>
/// Simple block sub-allocator: large blocks per usage, first-fit with coalescing, dedicated
/// allocations for big resources. Every block is backed by one VkBuffer with device-address usage,
/// so kernels only ever see raw addresses.
/// </summary>
internal sealed unsafe class MemoryAllocator : IDisposable
{
    const ulong BlockSize = 256UL << 20;
    const ulong Alignment = 256;
    readonly VulkanDevice _dev;
    readonly List<MemoryBlock> _blocks = [];
    readonly Lock _lock = new();
    readonly VkPhysicalDeviceMemoryProperties _props;

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
            if (size < BlockSize / 2)
            {
                foreach (var b in _blocks)
                {
                    if (b.Usage != usage || b.Dedicated) continue;
                    if (TryCarve(b, size, out var a)) return a;
                }
                var nb = NewBlock(BlockSize, usage, dedicated: false);
                if (TryCarve(nb, size, out var a2)) return a2;
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
                else i++;
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

    static bool TryCarve(MemoryBlock b, ulong size, out Allocation a)
    {
        foreach (var (off, sz) in b.Free)
        {
            if (sz < size) continue;
            b.Free.Remove(off);
            if (sz > size) b.Free.Add(off + size, sz - size);
            a = new Allocation
            {
                Buffer = b.Buffer, Offset = off, Size = size, Address = b.Address + off,
                Mapped = b.Mapped == null ? null : b.Mapped + off, Block = b, Coherent = b.Coherent,
            };
            return true;
        }
        a = null!;
        return false;
    }

    MemoryBlock NewBlock(ulong size, MemoryUsage usage, bool dedicated)
    {
        var api = _dev.Api;
        var bci = new VkBufferCreateInfo
        {
            size = size,
            usage = VkBufferUsageFlags.StorageBuffer | VkBufferUsageFlags.TransferSrc | VkBufferUsageFlags.TransferDst | VkBufferUsageFlags.ShaderDeviceAddress,
            sharingMode = VkSharingMode.Exclusive,
        };
        VkBuffer buf;
        api.vkCreateBuffer(&bci, null, &buf).CheckResult();
        VkMemoryRequirements req;
        api.vkGetBufferMemoryRequirements(buf, &req);

        var (typeIndex, flags) = PickType(req.memoryTypeBits, usage);
        var flagsInfo = new VkMemoryAllocateFlagsInfo { flags = VkMemoryAllocateFlags.DeviceAddress };
        var dedicatedInfo = new VkMemoryDedicatedAllocateInfo { buffer = buf, pNext = &flagsInfo };
        var mai = new VkMemoryAllocateInfo { pNext = dedicated ? &dedicatedInfo : &flagsInfo, allocationSize = req.size, memoryTypeIndex = typeIndex };
        VkDeviceMemory mem;
        var r = api.vkAllocateMemory(&mai, null, &mem);
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
            Memory = mem, Buffer = buf, Size = size, Address = api.vkGetBufferDeviceAddress(&bai), Mapped = mapped,
            Dedicated = dedicated, Usage = usage, Coherent = (flags & VkMemoryPropertyFlags.HostCoherent) != 0,
        };
        block.Free.Add(0, size);
        _blocks.Add(block);
        if ((flags & VkMemoryPropertyFlags.DeviceLocal) != 0) _dev.Track((long)req.size);
        return block;
    }

    const ulong VkWholeSize = ulong.MaxValue;

    (uint, VkMemoryPropertyFlags) PickType(uint bits, MemoryUsage usage)
    {
        VkMemoryPropertyFlags[][] prefs = usage switch
        {
            MemoryUsage.Device => [[VkMemoryPropertyFlags.DeviceLocal]],
            MemoryUsage.Upload =>
            [
                [VkMemoryPropertyFlags.DeviceLocal | VkMemoryPropertyFlags.HostVisible | VkMemoryPropertyFlags.HostCoherent],
                [VkMemoryPropertyFlags.HostVisible | VkMemoryPropertyFlags.HostCoherent],
            ],
            _ =>
            [
                [VkMemoryPropertyFlags.HostVisible | VkMemoryPropertyFlags.HostCoherent | VkMemoryPropertyFlags.HostCached],
                [VkMemoryPropertyFlags.HostVisible | VkMemoryPropertyFlags.HostCoherent],
            ],
        };
        foreach (var want in prefs[0..])
            foreach (var w in want)
                for (uint i = 0; i < _props.memoryTypeCount; i++)
                {
                    var f = _props.memoryTypes[(int)i].propertyFlags;
                    if ((bits & (1u << (int)i)) != 0 && (f & w) == w)
                    {
                        // For pure device memory avoid host-visible types if a plain one exists.
                        if (usage == MemoryUsage.Device && (f & VkMemoryPropertyFlags.HostVisible) != 0 && HasPlainDeviceLocal(bits)) continue;
                        return (i, f);
                    }
                }
        throw new NotSupportedException($"No memory type for {usage}.");
    }

    bool HasPlainDeviceLocal(uint bits)
    {
        for (uint i = 0; i < _props.memoryTypeCount; i++)
        {
            var f = _props.memoryTypes[(int)i].propertyFlags;
            if ((bits & (1u << (int)i)) != 0 && (f & VkMemoryPropertyFlags.DeviceLocal) != 0 && (f & VkMemoryPropertyFlags.HostVisible) == 0) return true;
        }
        return false;
    }

    void Destroy(MemoryBlock b)
    {
        if (b.Mapped != null) _dev.Api.vkUnmapMemory(b.Memory);
        _dev.Api.vkDestroyBuffer(b.Buffer, null);
        _dev.Api.vkFreeMemory(b.Memory, null);
        _dev.Track(-(long)b.Size);
    }

    public void Dispose()
    {
        lock (_lock)
        {
            foreach (var b in _blocks) Destroy(b);
            _blocks.Clear();
        }
    }
}
