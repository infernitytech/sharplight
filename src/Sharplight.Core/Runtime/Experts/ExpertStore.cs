using System.Diagnostics;
using Sharplight.Models;

namespace Sharplight.Runtime.Experts;

/// <summary>
/// Routed experts left on disk (<see cref="Tasks.TaskLoadOptions.ExpertsOnDisk"/>): when the experts a placement puts on the CPU do not fit host
/// memory, they stay in the model file and the CPU reads the ones a pass selects into a cache of fixed-size slots (one expert of one weight
/// each, least recently used evicted first) through an <see cref="IExpertReader"/> (io_uring on Linux, else positional reads on threads),
/// bypassing the page cache. The weights' bytes are used as stored, so the results equal those of the resident experts. One store serves
/// every compiled graph of a model; it decides once, from the first placement that puts experts on the CPU, whether they go to disk and how
/// large the cache is.
/// </summary>
/// <remarks>Thread-safe. Dispose it after the compiled models that use it.</remarks>
public sealed unsafe class ExpertStore : IDisposable
{
    /// <summary>Environment variable of the host memory kept free beside the cache, in MiB (default 4096).</summary>
    public const string HostReserveVariable = "SHARPLIGHT_EXPERT_HOST_RESERVE_MIB";

    /// <summary>Largest single read: larger experts are read in pieces, which keeps more of the device's queue busy.</summary>
    private const long MaxReadBytes = 8L << 20;

    /// <summary>The smallest cache the store sizes for itself.</summary>
    private const long MinCacheBytes = 1L << 30;

    /// <summary>Slots of a pool at least, whatever the budget.</summary>
    private const int MinPoolSlots = 64;

    /// <summary>Rows per pass that the CPU's fused small-pass path takes at once (all their experts are held together).</summary>
    private const int SmallPassRows = 4;

    /// <summary>Experts the read-speed probe reads at most.</summary>
    private const int ProbeExperts = 16;

    private readonly Lock _lock = new();
    private readonly IExpertReader _reader;
    private readonly bool _ownsReader;
    private readonly Action<string>? _log;
    private readonly Dictionary<(string Path, long Offset), DiskExpertTensor> _tensors = [];
    private readonly Dictionary<string, IExpertFile> _files = new(StringComparer.Ordinal);
    private readonly Dictionary<string, List<DiskExpertTensor>> _groups = new(StringComparer.Ordinal);
    private readonly List<ExpertSlotPool> _pools = [];
    /// <summary>Devices that copy from the cache slots (<see cref="PinFor"/>): page-lock and release a range of host memory.</summary>
    private readonly List<(object Device, Func<nint, long, bool> Pin, Action<nint> Unpin)> _pinners = [];
    /// <summary>The pools each device holds page-locked.</summary>
    private readonly Dictionary<object, HashSet<ExpertSlotPool>> _pinned = [];
    private bool _decided;
    private bool _layoutStale;
    private bool _estimateLogged;
    private long _clock;
    private long _hits;
    private long _misses;
    private long _prefetched;
    private long _readAhead;
    private long _evictions;
    private long _bytesRead;
    private long _readWaitTicks;
    private bool _disposed;

    /// <summary>Creates a store.</summary>
    /// <param name="cacheBytes">
    /// Bytes of host memory for the cache, or null to size it from the host's free memory. A size below the experts the placement puts on the
    /// CPU leaves them on disk even when they would fit.
    /// </param>
    /// <param name="reader">The reader, or null for <see cref="ExpertReaders.CreateDefault"/> (then owned by the store).</param>
    /// <param name="log">Receives the decision and the speed estimate, or null.</param>
    public ExpertStore(long? cacheBytes = null, IExpertReader? reader = null, Action<string>? log = null)
    {
        RequestedCacheBytes = cacheBytes;
        _log = log;
        _ownsReader = reader is null;
        _reader = reader ?? ExpertReaders.CreateDefault(log);
    }

    /// <summary>
    /// The store for a load with <see cref="Tasks.TaskLoadOptions.ExpertsOnDisk"/> and a CPU fallback device, or null otherwise (the experts then
    /// stay memory-mapped).
    /// </summary>
    internal static ExpertStore? ForLoad(Tasks.TaskLoadOptions options)
        => options.ExpertsOnDisk && options.Fallback is { Device.Info.Kind: DeviceKind.Cpu }
            ? new ExpertStore(options.ExpertCacheBytes, log: options.Log)
            : null;

    /// <summary>The cache size asked for, or null.</summary>
    public long? RequestedCacheBytes { get; }

    /// <summary>Whether experts were left on disk (decided by the first placement with experts on the CPU).</summary>
    public bool IsActive { get; private set; }

    /// <summary>The cache budget in bytes once <see cref="IsActive"/>.</summary>
    public long CacheBytes { get; private set; }

    /// <summary>The reader's name (<see cref="IExpertReader.Name"/>).</summary>
    public string ReaderName => _reader.Name;

    /// <summary>Bytes of the routed experts read from disk.</summary>
    public long DiskBackedBytes
    {
        get
        {
            lock (_lock)
            {
                return _tensors.Values.Sum(t => t.StoredExpertBytes * t.Experts);
            }
        }
    }

    /// <summary>Expert weights read from disk.</summary>
    public int DiskBackedWeights
    {
        get
        {
            lock (_lock)
            {
                return _tensors.Count;
            }
        }
    }

    /// <summary>Read throughput the store measured when it laid out its cache, in bytes per second, or null before that.</summary>
    public double? MeasuredReadBytesPerSecond { get; private set; }

    /// <summary>The counters since creation or <see cref="ResetStatistics"/>.</summary>
    public ExpertCacheStatistics Statistics
    {
        get
        {
            lock (_lock)
            {
                return new ExpertCacheStatistics(_hits, _readAhead, _misses, _prefetched, _evictions, _bytesRead, Stopwatch.GetElapsedTime(0, _readWaitTicks),
                    _pools.Sum(p => p.Bytes), _pools.Sum(p => p.Count));
            }
        }
    }

    /// <summary>Sets the counters of <see cref="Statistics"/> to zero (the cache keeps its contents).</summary>
    public void ResetStatistics()
    {
        lock (_lock)
        {
            _hits = 0;
            _misses = 0;
            _prefetched = 0;
            _readAhead = 0;
            _evictions = 0;
            _bytesRead = 0;
            _readWaitTicks = 0;
        }
    }

    /// <summary>
    /// Decides whether the routed experts a placement put on the CPU stay on disk: when the cache asked for is smaller than they are, or
    /// when they and the CPU's other weights and state do not fit <paramref name="hostAvailable"/> with the reserve kept free. The first
    /// decision to use the disk holds for the store's life (later placements put their CPU experts on disk too).
    /// </summary>
    /// <param name="cpuExpertBytes">Bytes of the routed experts on the CPU.</param>
    /// <param name="cpuOtherBytes">Bytes of the CPU's other weights and state.</param>
    /// <param name="hostAvailable">Host memory the process may still use.</param>
    internal bool Decide(long cpuExpertBytes, long cpuOtherBytes, long hostAvailable)
    {
        lock (_lock)
        {
            if (IsActive || cpuExpertBytes <= 0)
            {
                return IsActive;
            }

            long reserve = HostReserve;
            bool forced = RequestedCacheBytes is { } asked && asked < cpuExpertBytes;
            bool tooLarge = cpuExpertBytes + cpuOtherBytes + reserve > hostAvailable;
            if (!forced && !tooLarge)
            {
                if (!_decided)
                {
                    _decided = true;
                    _log?.Invoke($"Experts on disk: not needed, the {Gib(cpuExpertBytes)} of routed experts on the CPU fit host memory ({Gib(hostAvailable)} available).");
                }

                return false;
            }

            _decided = true;
            IsActive = true;
            CacheBytes = RequestedCacheBytes ?? Math.Clamp(hostAvailable - cpuOtherBytes - reserve, MinCacheBytes, Math.Max(MinCacheBytes, cpuExpertBytes));
            _log?.Invoke($"Experts on disk: {Gib(cpuExpertBytes)} of routed experts on the CPU "
                + (forced ? $"exceed the {Gib(CacheBytes)} cache asked for" : $"do not fit the {Gib(hostAvailable)} of host memory available")
                + $"; they stay in the model file and are read on demand ({_reader.Name}) into a {Gib(CacheBytes)} cache.");
            return true;
        }
    }

    /// <summary>Host memory kept free beside the cache (<see cref="HostReserveVariable"/>).</summary>
    internal static long HostReserve => long.TryParse(Environment.GetEnvironmentVariable(HostReserveVariable), out long mib) && mib >= 0 ? mib << 20 : 4L << 30;

    /// <summary>
    /// Leaves the expert weight <paramref name="weight"/> ([experts, N, K], one contiguous range of a file) on disk; false when it is not such a
    /// weight (it stays resident). Registering a weight again is a no-op.
    /// </summary>
    /// <param name="weight">The weight.</param>
    /// <param name="group">The placement group of its layer's routed experts.</param>
    /// <param name="slotsPerRow">Experts a row selects.</param>
    internal bool TryRegister(WeightInfo weight, string group, int slotsPerRow)
    {
        if (weight.Shape.Rank != 3 || weight.Shape.Dims.Any(d => d.IsSymbolic) || !TryLocate(weight, out string path, out long offset, out var encoding))
        {
            return false;
        }

        long experts = weight.Shape[0].Fixed;
        long bytes = weight.ByteSize;
        if (experts <= 0 || bytes % experts != 0 || (encoding is not null && (bytes / experts) % encoding.BlockBytes != 0))
        {
            return false;
        }

        lock (_lock)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_tensors.ContainsKey((path, offset)))
            {
                return true;
            }

            if (!_files.TryGetValue(path, out var file))
            {
                file = _reader.Open(path);
                _files[path] = file;
            }

            var tensor = new DiskExpertTensor(weight.Name, group, file, offset, bytes / experts, (int)experts, Math.Max(1, slotsPerRow), encoding);
            _tensors[(path, offset)] = tensor;
            if (!_groups.TryGetValue(group, out var members))
            {
                members = [];
                _groups[group] = members;
            }

            members.Add(tensor);
            _layoutStale = true;
            return true;
        }
    }

    /// <summary>
    /// Where <paramref name="weight"/>'s bytes lie in a file: as stored (<see cref="IWeightSource.TryGetFileRange"/>), or stored in another
    /// block encoding that the store decodes after each read (<see cref="IWeightSource.TryGetEncodedFileRange"/>).
    /// </summary>
    private static bool TryLocate(WeightInfo weight, out string path, out long offset, out IStoredBlockEncoding? encoding)
    {
        encoding = null;
        return weight.Source.TryGetFileRange(out path, out offset)
            || (weight.Source.TryGetEncodedFileRange(out path, out offset, out encoding) && encoding is not null);
    }

    /// <summary>The disk-backed form of <paramref name="weight"/>, or null when it is resident.</summary>
    internal DiskExpertTensor? Find(WeightInfo weight)
    {
        if (!TryLocate(weight, out string path, out long offset, out _))
        {
            return null;
        }

        lock (_lock)
        {
            return _tensors.GetValueOrDefault((path, offset));
        }
    }

    /// <summary>
    /// Lets a GPU copy experts straight from the cache slots: the store page-locks (imports) every pool for <paramref name="device"/> with
    /// <paramref name="pin"/> now and whenever it lays its pools out again, and releases them with <paramref name="unpin"/> before it frees them.
    /// Calling it again for the same device is a no-op. A pool the device's budget refuses stays pageable (<see cref="IsPinnedFor"/>).
    /// </summary>
    internal void PinFor(object device, Func<nint, long, bool> pin, Action<nint> unpin)
    {
        lock (_lock)
        {
            if (_pinners.Exists(p => p.Device == device))
            {
                return;
            }

            _pinners.Add((device, pin, unpin));
            _pinned[device] = [];
            EnsureLayout();
            foreach (var pool in _pools)
            {
                PinPoolLocked(device, pin, pool);
            }
        }
    }

    /// <summary>Whether <paramref name="tensor"/>'s pool is page-locked for <paramref name="device"/>.</summary>
    internal bool IsPinnedFor(object device, DiskExpertTensor tensor)
    {
        lock (_lock)
        {
            return tensor.Pool is { } pool && _pinned.TryGetValue(device, out var pools) && pools.Contains(pool);
        }
    }

    private void PinPoolLocked(object device, Func<nint, long, bool> pin, ExpertSlotPool pool)
    {
        if (pin((nint)pool.Arena, pool.Bytes))
        {
            _pinned[device].Add(pool);
        }
    }

    /// <summary>Releases a pool's page-locking on every device before the pool is freed. Holds the lock.</summary>
    private void UnpinPoolLocked(ExpertSlotPool pool)
    {
        foreach (var (device, _, unpin) in _pinners)
        {
            if (_pinned[device].Remove(pool))
            {
                unpin((nint)pool.Arena);
            }
        }
    }

    /// <summary>
    /// Experts the CPU should take at once from <paramref name="tensor"/>: a quarter of its pool, so that a pair's chunk and the next one read
    /// ahead fit together (and at least every expert a small pass selects: <see cref="MinSlots"/>).
    /// </summary>
    internal int ChunkExperts(DiskExpertTensor tensor)
    {
        lock (_lock)
        {
            EnsureLayout();
            return Math.Max(1, tensor.Pool!.Count / 4);
        }
    }

    /// <summary>Starts reading the experts of the weights that the cache does not hold, as far as slots that no pass holds allow.</summary>
    internal void Prefetch(DiskExpertTensor first, DiskExpertTensor? second, ReadOnlySpan<int> experts)
    {
        lock (_lock)
        {
            EnsureLayout();
            PrefetchLocked(first, experts);
            if (second is not null)
            {
                PrefetchLocked(second, experts);
            }
        }
    }

    private void PrefetchLocked(DiskExpertTensor tensor, ReadOnlySpan<int> experts)
    {
        var pool = tensor.Pool!;
        var missing = new List<(int Expert, int Slot)>();
        foreach (int e in experts)
        {
            if (tensor.SlotOf[e] >= 0)
            {
                continue;
            }

            int slot = pool.Take(out bool evicted);
            if (slot < 0)
            {
                break;
            }

            _evictions += evicted ? 1 : 0;
            pool[slot] = new ExpertSlotPool.Slot { Tensor = tensor, Expert = e, LastUse = ++_clock };
            tensor.SlotOf[e] = slot;
            missing.Add((e, slot));
        }

        if (missing.Count > 0)
        {
            _prefetched += missing.Count;
            SubmitLocked(tensor, missing);
        }
    }

    /// <summary>
    /// Holds the experts <paramref name="experts"/> (distinct ids) of <paramref name="first"/> (and <paramref name="second"/>) in the cache, reading
    /// those it lacks, and writes the address of each one's first byte to <paramref name="firstPointers"/>[expert] (and
    /// <paramref name="secondPointers"/>). With <paramref name="prefetchLayer"/> the same experts of the layer's other weights (its later
    /// products) are queued for reading behind these, so their reads overlap this pass's compute. They stay until <see cref="Release"/>.
    /// </summary>
    /// <exception cref="InvalidOperationException">The cache has too few free slots for the experts.</exception>
    /// <exception cref="IOException">A read failed.</exception>
    internal void Acquire(DiskExpertTensor first, DiskExpertTensor? second, ReadOnlySpan<int> experts, byte** firstPointers, byte** secondPointers, bool prefetchLayer)
    {
        long started = Stopwatch.GetTimestamp();
        var waits = new List<ExpertReadBatch>();
        lock (_lock)
        {
            EnsureLayout();
            PinLocked(first, experts, waits);
            if (second is not null)
            {
                try
                {
                    PinLocked(second, experts, waits);
                }
                catch (InvalidOperationException)
                {
                    UnpinLocked(first, experts);
                    throw;
                }
            }

            if (prefetchLayer)
            {
                foreach (var member in _groups[first.Group])
                {
                    if (member != first && member != second)
                    {
                        PrefetchLocked(member, experts);
                    }
                }
            }
        }

        foreach (var batch in waits)
        {
            try
            {
                batch.Wait();
            }
            catch (IOException)
            {
                Release(first, second, experts);
                throw;
            }
        }

        lock (_lock)
        {
            Resolve(first, experts, firstPointers);
            if (second is not null)
            {
                Resolve(second, experts, secondPointers);
            }

            _readWaitTicks += Stopwatch.GetTimestamp() - started;
        }
    }

    /// <summary>Pins the experts of one weight, taking slots for (and submitting the reads of) those not cached. Holds the lock.</summary>
    private void PinLocked(DiskExpertTensor tensor, ReadOnlySpan<int> experts, List<ExpertReadBatch> waits)
    {
        var pool = tensor.Pool!;
        var missing = new List<(int Expert, int Slot)>();
        foreach (int e in experts)
        {
            int slot = tensor.SlotOf[e];
            if (slot >= 0 && pool[slot].Loading is { IsCompleted: true, IsFaulted: true })
            {
                // A prefetch of this expert failed: read it again (the pass then sees the error if it persists).
                pool[slot].Loading = null;
                if (pool[slot].Pins == 0)
                {
                    pool.Forget(slot);
                    slot = -1;
                }
            }

            if (slot >= 0)
            {
                ref var held = ref pool[slot];
                if (held.Uses == 0)
                {
                    _readAhead++;
                }
                else
                {
                    _hits++;
                }

                held.Pins++;
                held.Uses++;
                held.LastUse = ++_clock;
                if (held.Loading is { } loading && !waits.Contains(loading))
                {
                    waits.Add(loading);
                }

                continue;
            }

            slot = pool.Take(out bool evicted);
            if (slot < 0)
            {
                // Undo: the experts taken so far are no longer held, and the slots taken for reads that never started are free again.
                foreach (int taken in experts)
                {
                    if (taken == e)
                    {
                        break;
                    }

                    int takenSlot = tensor.SlotOf[taken];
                    if (missing.Exists(m => m.Slot == takenSlot))
                    {
                        pool.Forget(takenSlot);
                    }
                    else
                    {
                        pool[takenSlot].Pins--;
                    }
                }

                throw new InvalidOperationException($"The expert cache of {pool.Count} slots for '{tensor.Name}' has no free slot for {experts.Length} experts at once.");
            }

            _misses++;
            _evictions += evicted ? 1 : 0;
            pool[slot] = new ExpertSlotPool.Slot { Tensor = tensor, Expert = e, Pins = 1, Uses = 1, LastUse = ++_clock };
            tensor.SlotOf[e] = slot;
            missing.Add((e, slot));
        }

        if (missing.Count > 0)
        {
            waits.Add(SubmitLocked(tensor, missing));
        }
    }

    /// <summary>Writes the cached experts' addresses. Holds the lock.</summary>
    private static void Resolve(DiskExpertTensor tensor, ReadOnlySpan<int> experts, byte** pointers)
    {
        var pool = tensor.Pool!;
        foreach (int e in experts)
        {
            int slot = tensor.SlotOf[e];
            ref var held = ref pool[slot];
            if (held.Loading is { IsCompleted: true })
            {
                held.Loading = null;
            }

            pointers[e] = pool.Address(slot) + tensor.DataOffset(e);
        }
    }

    /// <summary>Lets the cache evict experts held by <see cref="Acquire"/> again.</summary>
    internal void Release(DiskExpertTensor first, DiskExpertTensor? second, ReadOnlySpan<int> experts)
    {
        lock (_lock)
        {
            UnpinLocked(first, experts);
            if (second is not null)
            {
                UnpinLocked(second, experts);
            }
        }
    }

    private static void UnpinLocked(DiskExpertTensor tensor, ReadOnlySpan<int> experts)
    {
        var pool = tensor.Pool!;
        foreach (int e in experts)
        {
            int slot = tensor.SlotOf[e];
            if (slot >= 0 && pool[slot].Pins > 0)
            {
                pool[slot].Pins--;
            }
        }
    }

    /// <summary>Submits the reads that fill <paramref name="missing"/>, in pieces of at most <see cref="MaxReadBytes"/>. Holds the lock.</summary>
    private ExpertReadBatch SubmitLocked(DiskExpertTensor tensor, List<(int Expert, int Slot)> missing)
    {
        var pool = tensor.Pool!;
        var reads = new List<ExpertRead>();
        foreach (var (e, slot) in missing)
        {
            var (readOffset, _, readBytes) = tensor.ReadRange(e);
            byte* destination = pool.Address(slot);
            for (long done = 0; done < readBytes; done += MaxReadBytes)
            {
                int length = (int)Math.Min(MaxReadBytes, readBytes - done);
                reads.Add(new ExpertRead(tensor.File, readOffset + done, length, ExpertReadDestination.Host((nint)(destination + done))));
            }

            _bytesRead += readBytes;
        }

        var batch = _reader.Submit(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(reads));
        if (tensor.Encoding is not null)
        {
            batch = DecodeAfter(batch, tensor, missing);
        }

        foreach (var (_, slot) in missing)
        {
            pool[slot].Loading = batch;
        }

        return batch;
    }

    /// <summary>
    /// For a weight stored in another block encoding: a batch that completes once <paramref name="reads"/> did and the experts it filled are
    /// decoded in their slots (in parallel, outside the store's lock), so that a pass waiting for the slot finds the weight's own bytes.
    /// </summary>
    private static ExpertReadBatch DecodeAfter(ExpertReadBatch reads, DiskExpertTensor tensor, List<(int Expert, int Slot)> missing)
    {
        var pool = tensor.Pool!;
        var slots = missing.Select(m => (m.Expert, Address: (nint)pool.Address(m.Slot))).ToArray();
        var decoded = new ExpertReadBatch(1);
        Task.Run(() =>
        {
            try
            {
                reads.Wait();
                Parallel.For(0, slots.Length, i => tensor.Decode(slots[i].Expert, (byte*)slots[i].Address));
                decoded.CompleteRead(reads.BytesRead);
            }
            catch (Exception error)
            {
                decoded.FailRead(error is IOException { InnerException: { } inner } ? inner : error);
            }
        });
        return decoded;
    }

    /// <summary>
    /// Lays the cache out for the registered weights when they changed: one pool per slot size, the budget split by the bytes of the weights of
    /// each size (every weight of a layer is read as often), at least enough slots per pool for a few passes. Holds the lock.
    /// </summary>
    private void EnsureLayout()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (!_layoutStale)
        {
            return;
        }

        _layoutStale = false;
        var tensors = _tensors.Values.ToList();
        if (_pools.Any(p => p.AnyPinned))
        {
            // A pass holds slots: the weights registered since get pools of their own at the smallest size.
            foreach (var size in tensors.Where(t => t.Pool is null).GroupBy(t => t.SlotBytes))
            {
                var pool = new ExpertSlotPool(size.Key, MinSlots(size));
                _pools.Add(pool);
                foreach (var (device, pin, _) in _pinners)
                {
                    PinPoolLocked(device, pin, pool);
                }

                foreach (var t in size)
                {
                    t.Pool = pool;
                }
            }

            return;
        }

        foreach (var pool in _pools)
        {
            pool.WaitForReads();
            UnpinPoolLocked(pool);
            pool.Dispose();
        }

        _pools.Clear();
        long budget = CacheBytes > 0 ? CacheBytes : RequestedCacheBytes ?? MinCacheBytes;
        double totalWeight = tensors.Sum(t => (double)t.SlotBytes);
        foreach (var size in tensors.GroupBy(t => t.SlotBytes))
        {
            int count = size.Count();
            long share = (long)(budget * (count * (double)size.Key / totalWeight));
            long slots = Math.Min(Math.Max(share / size.Key, MinSlots(size)), size.Sum(t => (long)t.Experts));
            var pool = new ExpertSlotPool(size.Key, (int)slots);
            _pools.Add(pool);
            foreach (var (device, pin, _) in _pinners)
            {
                PinPoolLocked(device, pin, pool);
            }

            foreach (var t in size)
            {
                t.Pool = pool;
                Array.Fill(t.SlotOf, -1);
            }
        }

        if (!_estimateLogged)
        {
            _estimateLogged = true;
            ProbeAndEstimate(tensors);
        }
    }

    /// <summary>
    /// The smallest pool for weights of one slot size: a small pass's experts (all held at once, for a gate / up pair twice) four times over,
    /// so that <see cref="ChunkExperts"/> takes a small pass whole.
    /// </summary>
    private static int MinSlots(IEnumerable<DiskExpertTensor> tensors)
        => Math.Max(MinPoolSlots, 2 * 4 * SmallPassRows * tensors.Max(t => t.SlotsPerRow));

    /// <summary>Reads a few experts to measure the disk and logs what decode speed to expect. Holds the lock.</summary>
    private void ProbeAndEstimate(List<DiskExpertTensor> tensors)
    {
        var random = new Random(1);
        long bytes = 0;
        long started = Stopwatch.GetTimestamp();
        var batches = new List<(DiskExpertTensor Tensor, int Expert, int Slot, ExpertReadBatch Batch)>();
        for (int i = 0; i < ProbeExperts && tensors.Count > 0; i++)
        {
            var tensor = tensors[random.Next(tensors.Count)];
            int e = random.Next(tensor.Experts);
            if (tensor.SlotOf[e] >= 0)
            {
                continue;
            }

            int slot = tensor.Pool!.Take(out _);
            if (slot < 0)
            {
                break;
            }

            tensor.Pool[slot] = new ExpertSlotPool.Slot { Tensor = tensor, Expert = e, LastUse = ++_clock };
            tensor.SlotOf[e] = slot;
            var batch = SubmitLocked(tensor, [(e, slot)]);
            batches.Add((tensor, e, slot, batch));
            bytes += tensor.ReadRange(e).ReadBytes;
        }

        bool failed = false;
        foreach (var (tensor, _, slot, batch) in batches)
        {
            try
            {
                batch.Wait();
                tensor.Pool![slot].Loading = null;
            }
            catch (IOException e)
            {
                failed = true;
                tensor.Pool!.Forget(slot);
                _log?.Invoke($"Experts on disk: a probe read of '{tensor.Name}' failed: {e.Message}");
            }
        }

        double seconds = Stopwatch.GetElapsedTime(started).TotalSeconds;
        _bytesRead = 0;
        if (failed || bytes == 0 || seconds <= 0)
        {
            return;
        }

        MeasuredReadBytesPerSecond = bytes / seconds;
        long diskBytes = tensors.Sum(t => t.StoredExpertBytes * t.Experts);
        long perToken = tensors.Sum(t => t.StoredExpertBytes * t.SlotsPerRow);
        double cachedShare = Math.Clamp(_pools.Sum(p => (double)p.Bytes) / diskBytes, 0, 1);
        double coldSeconds = perToken / MeasuredReadBytesPerSecond.Value;
        double warmSeconds = coldSeconds * (1 - cachedShare);
        _log?.Invoke($"Experts on disk: {tensors.Count} weights in {_groups.Count} layers ({Gib(diskBytes)}) read with {_reader.Name}; "
            + $"{bytes / seconds / 1e9:F1} GB/s measured; a decode step reads up to {perToken / 1e9:F2} GB: about {1 / coldSeconds:F2} tok/s from a cold cache, "
            + (warmSeconds > 0 ? $"~{1 / warmSeconds:F2} tok/s once the {_pools.Sum(p => p.Bytes) / (double)(1L << 30):F1} GiB cache holds its share (reads only; compute adds to it)." : "all of it cached once warm."));
    }

    private static string Gib(long bytes) => $"{bytes / (double)(1L << 30):F1} GiB";

    /// <inheritdoc/>
    public void Dispose()
    {
        lock (_lock)
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            foreach (var pool in _pools)
            {
                pool.WaitForReads();
            }
        }

        if (_ownsReader)
        {
            _reader.Dispose();
        }

        foreach (var pool in _pools)
        {
            UnpinPoolLocked(pool);
            pool.Dispose();
        }

        foreach (var file in _files.Values)
        {
            file.Dispose();
        }
    }
}
