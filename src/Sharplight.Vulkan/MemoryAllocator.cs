using System.Globalization;
using System.Text;
using Sharplight.Runtime;
using Vortice.Vulkan;

namespace Sharplight.Vulkan;

/// <summary>
/// Block sub-allocator: blocks per usage, first-fit with coalescing, dedicated allocations for big resources. Every block is backed
/// by one VkBuffer with device-address usage, so kernels only ever see raw addresses.
/// </summary>
/// <remarks>
/// Nothing about the memory topology is assumed. Memory types are tried in order of preference and the next one serves when a heap
/// is full (staging in device-local host-visible memory, the PCIe BAR window, falls back to host memory). A block that cannot be
/// allocated whole is replaced by an allocation just large enough. Blocks never exceed the largest allocation the driver accepts, and
/// a buffer larger than that is bound to several allocations (sparse binding) where the device can. Caps of
/// <see cref="DeviceMemoryLimits"/> act like smaller heaps. Only when nothing works does an allocation fail, with a
/// <see cref="DeviceOutOfMemoryException"/> naming the allocation, the attempts and every heap's figures.
/// </remarks>
internal sealed unsafe class MemoryAllocator : IDisposable
{
    /// <summary>Weights and state go in big device blocks; the small host-visible staging buffers of inputs/outputs in small ones (a 256 MiB mapped block per usage wastes memory on small models).</summary>
    private const ulong DeviceBlockSize = 256UL << 20;
    private const ulong StagingBlockSize = 16UL << 20;
    private const ulong Alignment = 256;
    private const ulong VkWholeSize = ulong.MaxValue;
    private readonly VulkanDevice _dev;
    private readonly List<MemoryBlock> _blocks = [];
    private readonly Lock _lock = new();
    private bool _disposed;
    private readonly VkPhysicalDeviceMemoryProperties _props;
    /// <summary>Bytes allocated from each heap by this allocator.</summary>
    private readonly ulong[] _heapUsed;
    /// <summary>The device has device-local memory the host cannot map: device-local host-visible types are a separate (BAR) window.</summary>
    private readonly bool _discrete;
    private readonly long? _deviceMemoryCap;
    private readonly long? _hostVisibleDeviceCap;
    private ulong _hostVisibleDeviceUsed;

    public MemoryAllocator(VulkanDevice dev)
    {
        _dev = dev;
        VkPhysicalDeviceMemoryProperties mp;
        dev.Inst.Api.vkGetPhysicalDeviceMemoryProperties(dev.Phys.Handle, &mp);
        _props = mp;
        _heapUsed = new ulong[mp.memoryHeapCount];
        _discrete = HasPlainDeviceLocal(uint.MaxValue);
        _deviceMemoryCap = dev.Limits.DeviceMemory;
        _hostVisibleDeviceCap = dev.Limits.HostVisibleDeviceMemory;
        ulong driverMax = Math.Min(dev.Phys.MaxMemoryAllocationSize, long.MaxValue);
        ulong max = (ulong)DeviceMemoryLimits.Apply((long)driverMax, dev.Limits.MaxAllocation);
        MaxAllocation = Math.Max(Alignment, max & ~(Alignment - 1));
    }

    /// <summary>The largest single allocation: the driver's limit or <see cref="DeviceMemoryLimits.MaxAllocation"/>, whichever is smaller.</summary>
    public ulong MaxAllocation { get; }

    /// <summary>Allocates <paramref name="size"/> bytes of <paramref name="usage"/> memory.</summary>
    /// <param name="size">Bytes.</param>
    /// <param name="usage">Where the memory should live.</param>
    /// <param name="purpose">What the memory is for (for the error message), e.g. "weight blk.0.attn_q.weight".</param>
    /// <exception cref="DeviceOutOfMemoryException">No memory type, block size or split could serve the request.</exception>
    public Allocation Allocate(ulong size, MemoryUsage usage, string? purpose = null)
    {
        size = Math.Max(Alignment, (size + Alignment - 1) & ~(Alignment - 1));
        lock (_lock)
        {
            ulong blockSize = Math.Min(usage == MemoryUsage.Device ? DeviceBlockSize : StagingBlockSize, MaxAllocation);
            var attempts = new List<string>();
            if (size < blockSize / 2)
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

                // A full block may not fit any more (a small heap, other processes): then a dedicated allocation of just this size.
                var nb = NewBlock(blockSize, usage, dedicated: false, attempts);
                if (nb is not null && TryCarve(nb, size, out var a2))
                {
                    return a2;
                }
            }

            if (size > _dev.Phys.MaxBufferSize)
            {
                throw OutOfMemory(size, usage, purpose,
                    $"The buffer exceeds the device's largest buffer ({Mib(_dev.Phys.MaxBufferSize)}).", attempts);
            }

            MemoryBlock? db;
            if (size <= MaxAllocation)
            {
                db = NewBlock(size, usage, dedicated: true, attempts);
            }
            else if (usage == MemoryUsage.Device && _dev.SparseBinding)
            {
                db = NewSparseBlock(size, attempts);
            }
            else
            {
                throw OutOfMemory(size, usage, purpose,
                    $"The buffer exceeds the largest allocation ({Mib(MaxAllocation)}) and the device cannot bind a buffer to several allocations (sparseBinding).", attempts);
            }

            if (db is null)
            {
                throw OutOfMemory(size, usage, purpose, "", attempts);
            }

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

    /// <summary>The property flags of memory type <paramref name="typeIndex"/>.</summary>
    internal VkMemoryPropertyFlags TypeFlags(uint typeIndex) => _props.memoryTypes[(int)typeIndex].propertyFlags;

    /// <summary>The live per-heap budget and usage (VK_EXT_memory_budget), or empty figures when the extension is not enabled.</summary>
    public MemoryHeapBudgets ReadHeapBudgets()
    {
        if (!_dev.MemoryBudgetEnabled)
        {
            return new MemoryHeapBudgets([], []);
        }

        var bp = new VkPhysicalDeviceMemoryBudgetPropertiesEXT();
        var mp = new VkPhysicalDeviceMemoryProperties2 { pNext = &bp };
        _dev.Inst.Api.vkGetPhysicalDeviceMemoryProperties2(_dev.Phys.Handle, &mp);
        var budget = new ulong[_props.memoryHeapCount];
        var usage = new ulong[_props.memoryHeapCount];
        for (int i = 0; i < budget.Length; i++)
        {
            budget[i] = bp.heapBudget[i];
            usage[i] = bp.heapUsage[i];
        }

        return new MemoryHeapBudgets(budget, usage);
    }

    /// <summary>
    /// Bytes of upload staging memory that can be allocated now without crowding the device: the free budget of the heap the preferred
    /// upload type lives on (VK_EXT_memory_budget), capped by <see cref="DeviceMemoryLimits.HostVisibleDeviceMemory"/> and the largest
    /// allocation; <see cref="MaxAllocation"/> when nothing else is known.
    /// </summary>
    public ulong StagingHeadroom()
    {
        lock (_lock)
        {
            var types = CandidateTypes(uint.MaxValue, MemoryUsage.Upload);
            if (types.Count == 0)
            {
                return 0;
            }

            var (index, flags) = types[0];
            ulong headroom = MaxAllocation;
            if (IsHostVisibleDevice(flags) && _hostVisibleDeviceCap is { } cap)
            {
                headroom = Math.Min(headroom, (ulong)Math.Max(0, cap - (long)_hostVisibleDeviceUsed));
            }

            var heaps = ReadHeapBudgets();
            if (heaps.IsKnown)
            {
                uint heap = _props.memoryTypes[(int)index].heapIndex;
                ulong free = heaps.Budget[heap] > heaps.Usage[heap] ? heaps.Budget[heap] - heaps.Usage[heap] : 0;
                headroom = Math.Min(headroom, free);
            }

            // When the preferred heap has no room, staging falls back to host memory: plenty of it.
            return headroom == 0 && types.Count > 1 ? MaxAllocation : headroom;
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

    private VkBuffer CreateBuffer(ulong size, VkBufferCreateFlags flags)
    {
        VkBuffer buf;
        // With a transfer queue, buffers are shared by both queue families (outputs are downloaded on the transfer queue).
        fixed (uint* families = _dev.SharingFamilies)
        {
            var bci = new VkBufferCreateInfo
            {
                flags = flags,
                size = size,
                usage = VkBufferUsageFlags.StorageBuffer | VkBufferUsageFlags.TransferSrc | VkBufferUsageFlags.TransferDst | VkBufferUsageFlags.ShaderDeviceAddress,
                sharingMode = _dev.SharingFamilies.Length > 1 ? VkSharingMode.Concurrent : VkSharingMode.Exclusive,
                queueFamilyIndexCount = (uint)_dev.SharingFamilies.Length,
                pQueueFamilyIndices = _dev.SharingFamilies.Length > 1 ? families : null,
            };
            _dev.Api.vkCreateBuffer(&bci, null, &buf).CheckResult();
        }

        return buf;
    }

    /// <summary>A block of one allocation, from the first memory type that can serve it; null (with the attempts recorded) when none can.</summary>
    private MemoryBlock? NewBlock(ulong size, MemoryUsage usage, bool dedicated, List<string> attempts)
    {
        var api = _dev.Api;
        var buf = CreateBuffer(size, 0);
        VkMemoryRequirements req;
        api.vkGetBufferMemoryRequirements(buf, &req);

        var flagsInfo = new VkMemoryAllocateFlagsInfo { flags = VkMemoryAllocateFlags.DeviceAddress };
        var dedicatedInfo = new VkMemoryDedicatedAllocateInfo { buffer = buf, pNext = &flagsInfo };
        VkDeviceMemory mem = default;
        uint chosen = uint.MaxValue;
        // Memory types in order of preference: when the preferred heap is full, the next type serves (staging in device-local
        // host-visible memory falls back to plain host memory, e.g. on GPUs without resizable BAR, whose 256 MiB BAR heap is
        // shared by every process on the device).
        var candidates = CandidateTypes(req.memoryTypeBits, usage);
        var heaps = usage == MemoryUsage.Device ? null : ReadHeapBudgets();
        for (int c = 0; c < candidates.Count; c++)
        {
            var (typeIndex, _) = candidates[c];
            // Staging has alternatives: leave a heap whose live budget is used up to the processes already using it.
            if (heaps is { IsKnown: true } && c + 1 < candidates.Count && !FitsBudget(heaps, typeIndex, req.size))
            {
                attempts.Add($"{Mib(size)} {usage} in type {typeIndex} ({Describe(typeIndex)}): skipped, the heap's budget is used up");
                continue;
            }

            var r = TryAllocateMemory(req.size, typeIndex, dedicated ? &dedicatedInfo : &flagsInfo, out mem);
            if (r == VkResult.Success)
            {
                chosen = typeIndex;
                break;
            }

            attempts.Add($"{Mib(size)} {usage} in type {typeIndex} ({Describe(typeIndex)}): {r}");
        }
        if (chosen == uint.MaxValue)
        {
            api.vkDestroyBuffer(buf, null);
            return null;
        }

        var flags = _props.memoryTypes[(int)chosen].propertyFlags;
        api.vkBindBufferMemory(buf, mem, 0).CheckResult();
        byte* mapped = null;
        if ((flags & VkMemoryPropertyFlags.HostVisible) != 0)
        {
            void* p;
            api.vkMapMemory(mem, 0, VkWholeSize, 0, &p).CheckResult();
            mapped = (byte*)p;
        }

        return AddBlock(buf, [new MemoryChunk(mem, chosen, req.size)], size, mapped, dedicated, usage, flags);
    }

    /// <summary>
    /// A device buffer larger than one allocation may be, bound to several allocations of at most <see cref="MaxAllocation"/> bytes
    /// (sparse binding): kernels still see one contiguous range of device addresses.
    /// </summary>
    private MemoryBlock? NewSparseBlock(ulong size, List<string> attempts)
    {
        var api = _dev.Api;
        if (size > _dev.Phys.SparseAddressSpaceSize)
        {
            attempts.Add($"sparse buffer of {Mib(size)}: larger than the sparse address space ({Mib(_dev.Phys.SparseAddressSpaceSize)})");
            return null;
        }

        var buf = CreateBuffer(size, VkBufferCreateFlags.SparseBinding);
        VkMemoryRequirements req;
        api.vkGetBufferMemoryRequirements(buf, &req);
        ulong chunkSize = MaxAllocation / req.alignment * req.alignment;
        if (chunkSize == 0)
        {
            api.vkDestroyBuffer(buf, null);
            attempts.Add($"sparse buffer of {Mib(size)}: the sparse block alignment ({req.alignment} bytes) exceeds the largest allocation");
            return null;
        }

        var chunks = new List<MemoryChunk>();
        var flagsInfo = new VkMemoryAllocateFlagsInfo { flags = VkMemoryAllocateFlags.DeviceAddress };
        var candidates = CandidateTypes(req.memoryTypeBits, MemoryUsage.Device);
        bool failed = false;
        for (ulong offset = 0; offset < req.size && !failed; offset += chunkSize)
        {
            ulong bytes = Math.Min(chunkSize, req.size - offset);
            failed = true;
            foreach (var (typeIndex, _) in candidates)
            {
                var r = TryAllocateMemory(bytes, typeIndex, &flagsInfo, out var mem);
                if (r == VkResult.Success)
                {
                    chunks.Add(new MemoryChunk(mem, typeIndex, bytes));
                    failed = false;
                    break;
                }

                attempts.Add($"sparse part {chunks.Count} of {Mib(bytes)} in type {typeIndex} ({Describe(typeIndex)}): {r}");
            }
        }

        if (failed)
        {
            api.vkDestroyBuffer(buf, null);
            foreach (var chunk in chunks)
            {
                FreeMemory(chunk);
            }

            return null;
        }

        var binds = new VkSparseMemoryBind[chunks.Count];
        ulong resourceOffset = 0;
        for (int i = 0; i < chunks.Count; i++)
        {
            binds[i] = new VkSparseMemoryBind { resourceOffset = resourceOffset, size = chunks[i].Size, memory = chunks[i].Memory, memoryOffset = 0 };
            resourceOffset += chunks[i].Size;
        }

        fixed (VkSparseMemoryBind* pBinds = binds)
        {
            var bufferBind = new VkSparseBufferMemoryBindInfo { buffer = buf, bindCount = (uint)binds.Length, pBinds = pBinds };
            var info = new VkBindSparseInfo { bufferBindCount = 1, pBufferBinds = &bufferBind };
            _dev.BindSparseAndWait(&info);
        }

        var flags = _props.memoryTypes[(int)chunks[0].TypeIndex].propertyFlags;
        return AddBlock(buf, chunks, size, null, dedicated: true, MemoryUsage.Device, flags);
    }

    private MemoryBlock AddBlock(VkBuffer buf, IReadOnlyList<MemoryChunk> chunks, ulong size, byte* mapped, bool dedicated, MemoryUsage usage, VkMemoryPropertyFlags flags)
    {
        var bai = new VkBufferDeviceAddressInfo { buffer = buf };
        long tracked = (flags & VkMemoryPropertyFlags.DeviceLocal) != 0 ? chunks.Sum(c => (long)c.Size) : 0;
        var block = new MemoryBlock
        {
            Memory = chunks,
            Buffer = buf,
            Size = size,
            Address = _dev.Api.vkGetBufferDeviceAddress(&bai),
            Mapped = mapped,
            Dedicated = dedicated,
            Usage = usage,
            Coherent = (flags & VkMemoryPropertyFlags.HostCoherent) != 0,
            Tracked = tracked,
        };
        block.Free.Add(0, size);
        _blocks.Add(block);
        _dev.Track(block.Tracked);
        return block;
    }

    /// <summary>vkAllocateMemory, after checking the caps of <see cref="DeviceMemoryLimits"/> (a capped heap fails as a full one would).</summary>
    private VkResult TryAllocateMemory(ulong size, uint typeIndex, void* pNext, out VkDeviceMemory memory)
    {
        memory = default;
        if (size > MaxAllocation)
        {
            return VkResult.ErrorOutOfDeviceMemory;
        }

        var flags = _props.memoryTypes[(int)typeIndex].propertyFlags;
        uint heap = _props.memoryTypes[(int)typeIndex].heapIndex;
        bool deviceHeap = (_props.memoryHeaps[(int)heap].flags & VkMemoryHeapFlags.DeviceLocal) != 0;
        if (deviceHeap && _deviceMemoryCap is { } deviceCap && (long)(DeviceHeapsUsed() + size) > deviceCap)
        {
            return VkResult.ErrorOutOfDeviceMemory;
        }

        bool hostVisibleDevice = IsHostVisibleDevice(flags);
        if (hostVisibleDevice && _hostVisibleDeviceCap is { } windowCap && (long)(_hostVisibleDeviceUsed + size) > windowCap)
        {
            return VkResult.ErrorOutOfDeviceMemory;
        }

        var mai = new VkMemoryAllocateInfo { pNext = pNext, allocationSize = size, memoryTypeIndex = typeIndex };
        VkDeviceMemory mem;
        var r = _dev.Api.vkAllocateMemory(&mai, null, &mem);
        if (r != VkResult.Success)
        {
            if (r is not (VkResult.ErrorOutOfDeviceMemory or VkResult.ErrorOutOfHostMemory or VkResult.ErrorTooManyObjects))
            {
                r.CheckResult();
            }

            return r;
        }

        memory = mem;
        _heapUsed[heap] += size;
        if (hostVisibleDevice)
        {
            _hostVisibleDeviceUsed += size;
        }

        return VkResult.Success;
    }

    private void FreeMemory(MemoryChunk chunk)
    {
        _dev.Api.vkFreeMemory(chunk.Memory, null);
        _heapUsed[_props.memoryTypes[(int)chunk.TypeIndex].heapIndex] -= chunk.Size;
        if (IsHostVisibleDevice(_props.memoryTypes[(int)chunk.TypeIndex].propertyFlags))
        {
            _hostVisibleDeviceUsed -= chunk.Size;
        }
    }

    private ulong DeviceHeapsUsed()
    {
        ulong used = 0;
        for (int i = 0; i < _heapUsed.Length; i++)
        {
            if ((_props.memoryHeaps[i].flags & VkMemoryHeapFlags.DeviceLocal) != 0)
            {
                used += _heapUsed[i];
            }
        }

        return used;
    }

    /// <summary>Device-local memory the host can map, on a device that also has device-local memory it cannot (the BAR window of a discrete GPU).</summary>
    private bool IsHostVisibleDevice(VkMemoryPropertyFlags flags)
    {
        const VkMemoryPropertyFlags window = VkMemoryPropertyFlags.DeviceLocal | VkMemoryPropertyFlags.HostVisible;
        return _discrete && (flags & window) == window;
    }

    private bool FitsBudget(MemoryHeapBudgets heaps, uint typeIndex, ulong size)
    {
        uint heap = _props.memoryTypes[(int)typeIndex].heapIndex;
        return heaps.Usage[heap] + size <= heaps.Budget[heap];
    }

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

                // A zero cap hides the BAR window, as on GPUs without one.
                if (IsHostVisibleDevice(f) && _hostVisibleDeviceCap == 0)
                {
                    continue;
                }

                types.Add((i, f));
            }
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

    private string Describe(uint typeIndex)
    {
        var f = _props.memoryTypes[(int)typeIndex].propertyFlags;
        var names = new List<string>();
        if ((f & VkMemoryPropertyFlags.DeviceLocal) != 0)
        {
            names.Add("device-local");
        }

        if ((f & VkMemoryPropertyFlags.HostVisible) != 0)
        {
            names.Add("host-visible");
        }

        if ((f & VkMemoryPropertyFlags.HostCached) != 0)
        {
            names.Add("host-cached");
        }

        return $"{string.Join(" ", names)}, heap {_props.memoryTypes[(int)typeIndex].heapIndex}";
    }

    private DeviceOutOfMemoryException OutOfMemory(ulong size, MemoryUsage usage, string? purpose, string reason, List<string> attempts)
    {
        var details = new StringBuilder();
        if (reason.Length > 0)
        {
            details.Append(reason).Append(' ');
        }

        if (attempts.Count > 0)
        {
            details.Append("Tried: ").Append(string.Join("; ", attempts)).Append(". ");
        }

        var heaps = ReadHeapBudgets();
        details.Append("Heaps:");
        for (int i = 0; i < _heapUsed.Length; i++)
        {
            var heap = _props.memoryHeaps[i];
            string kind = (heap.flags & VkMemoryHeapFlags.DeviceLocal) != 0 ? "device-local" : "host";
            details.Append(CultureInfo.InvariantCulture, $" [{i}] {kind} {Mib(heap.size)}");
            if (heaps.IsKnown)
            {
                details.Append(CultureInfo.InvariantCulture, $", budget {Mib(heaps.Budget[i])}, process usage {Mib(heaps.Usage[i])}");
            }

            details.Append(CultureInfo.InvariantCulture, $", Sharplight {Mib(_heapUsed[i])};");
        }

        details.Append(CultureInfo.InvariantCulture, $" Largest allocation {Mib(MaxAllocation)}.");
        if (!_dev.Limits.IsEmpty)
        {
            details.Append(CultureInfo.InvariantCulture, $" Limits: {_dev.Limits}.");
        }

        return new DeviceOutOfMemoryException(_dev.Info.Name, purpose ?? $"{usage} memory", (long)size, details.ToString());
    }

    private static string Mib(ulong bytes) => bytes == ulong.MaxValue ? "unlimited" : string.Create(CultureInfo.InvariantCulture, $"{bytes / 1048576.0:0.#} MiB");

    private void Destroy(MemoryBlock b)
    {
        if (b.Mapped != null)
        {
            _dev.Api.vkUnmapMemory(b.Memory[0].Memory);
        }

        _dev.Api.vkDestroyBuffer(b.Buffer, null);
        foreach (var chunk in b.Memory)
        {
            FreeMemory(chunk);
        }

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
