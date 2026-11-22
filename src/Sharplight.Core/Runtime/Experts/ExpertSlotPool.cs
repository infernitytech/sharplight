using System.Runtime.InteropServices;

namespace Sharplight.Runtime.Experts;

/// <summary>
/// Cache slots of one size (<see cref="DiskExpertTensor.SlotBytes"/>) in one page-aligned allocation. Eviction takes only slots that no pass
/// holds and no read fills, by segmented LRU: experts used at least twice are protected (up to <see cref="ProtectedShare"/> of the slots), the
/// least recently used of the others goes first, so that a prefill, which touches every expert once, does not flush the experts decode keeps
/// reusing. <c>SHARPLIGHT_EXPERT_CACHE=lru</c> evicts plain least recently used first. Not thread-safe: the <see cref="ExpertStore"/> lock guards it.
/// </summary>
internal sealed unsafe class ExpertSlotPool : IDisposable
{
    /// <summary>Most of the slots the protected segment may hold before its least recently used expert is evicted instead.</summary>
    private const double ProtectedShare = 0.8;

    /// <summary>Whether eviction is segmented (default) or plain least recently used.</summary>
    private static readonly bool s_segmented = Environment.GetEnvironmentVariable("SHARPLIGHT_EXPERT_CACHE") is not "lru";

    private readonly Slot[] _slots;
    private readonly Stack<int> _free = new();

    /// <summary>The state of one slot.</summary>
    internal struct Slot
    {
        /// <summary>The tensor whose expert the slot holds, or null.</summary>
        public DiskExpertTensor? Tensor;
        /// <summary>The expert.</summary>
        public int Expert;
        /// <summary>Passes holding the slot (<see cref="ExpertStore.Acquire"/> until <see cref="ExpertStore.Release"/>).</summary>
        public int Pins;
        /// <summary>The store's clock at the last use.</summary>
        public long LastUse;
        /// <summary>Passes that used the expert since it was read (0 for a prefetch nobody used yet).</summary>
        public int Uses;
        /// <summary>The batch filling the slot, or null once it is filled.</summary>
        public ExpertReadBatch? Loading;
    }

    public ExpertSlotPool(long slotBytes, int count)
    {
        SlotBytes = slotBytes;
        Count = count;
        Arena = (byte*)NativeMemory.AlignedAlloc((nuint)(slotBytes * count), ExpertFileHandle.DirectAlignment);
        _slots = new Slot[count];
        for (int i = count - 1; i >= 0; i--)
        {
            _free.Push(i);
        }
    }

    /// <summary>Bytes per slot.</summary>
    public long SlotBytes { get; }

    /// <summary>Slots in the pool.</summary>
    public int Count { get; }

    /// <summary>The first slot's memory.</summary>
    public byte* Arena { get; private set; }

    /// <summary>Bytes held.</summary>
    public long Bytes => SlotBytes * Count;

    /// <summary>The slot's state.</summary>
    public ref Slot this[int slot] => ref _slots[slot];

    /// <summary>The first byte of a slot.</summary>
    public byte* Address(int slot) => Arena + slot * SlotBytes;

    /// <summary>
    /// A slot to fill: a free one, else the least recently used one that is neither held nor being filled (its expert forgotten), or -1 when
    /// every slot is held or being filled.
    /// </summary>
    /// <param name="evicted">Whether a cached expert was evicted.</param>
    public int Take(out bool evicted)
    {
        evicted = false;
        if (_free.Count > 0)
        {
            return _free.Pop();
        }

        int best = -1;
        long oldest = long.MaxValue;
        int bestProbation = -1;
        long oldestProbation = long.MaxValue;
        int bestProtected = -1;
        long oldestProtected = long.MaxValue;
        int protectedCount = 0;
        for (int i = 0; i < _slots.Length; i++)
        {
            ref var s = ref _slots[i];
            if (s.Loading is { IsCompleted: true } done)
            {
                // A prefetch that no pass waited for: filled, or failed (then its expert is forgotten and the slot is the first to take).
                s.Loading = null;
                if (done.IsFaulted && s.Pins == 0)
                {
                    if (s.Tensor is { } failed)
                    {
                        failed.SlotOf[s.Expert] = -1;
                    }

                    s = default;
                    s.LastUse = long.MinValue;
                }
            }

            bool protectedSlot = s.Uses >= 2;
            protectedCount += protectedSlot ? 1 : 0;
            if (s.Pins != 0 || s.Loading is not null)
            {
                continue;
            }

            if (s.LastUse < oldest)
            {
                oldest = s.LastUse;
                best = i;
            }

            if (protectedSlot && s.LastUse < oldestProtected)
            {
                oldestProtected = s.LastUse;
                bestProtected = i;
            }
            else if (!protectedSlot && s.LastUse < oldestProbation)
            {
                oldestProbation = s.LastUse;
                bestProbation = i;
            }
        }

        if (s_segmented && best >= 0)
        {
            bool overfull = protectedCount > ProtectedShare * _slots.Length;
            best = overfull && bestProtected >= 0 ? bestProtected : bestProbation >= 0 ? bestProbation : best;
        }

        if (best >= 0)
        {
            ref var victim = ref _slots[best];
            if (victim.Tensor is { } tensor)
            {
                tensor.SlotOf[victim.Expert] = -1;
                evicted = true;
            }

            victim = default;
        }

        return best;
    }

    /// <summary>Forgets a slot's expert (a failed read) and returns the slot to the free list.</summary>
    public void Forget(int slot)
    {
        ref var s = ref _slots[slot];
        if (s.Tensor is { } tensor)
        {
            tensor.SlotOf[s.Expert] = -1;
        }

        s = default;
        _free.Push(slot);
    }

    /// <summary>Waits for every read still filling a slot (before the pool is freed).</summary>
    public void WaitForReads()
    {
        foreach (var s in _slots)
        {
            if (s.Loading is { } batch)
            {
                try
                {
                    batch.Wait();
                }
                catch (IOException)
                {
                    // The pool goes away; its contents do not matter any more.
                }
            }
        }
    }

    /// <summary>Whether a pass holds a slot.</summary>
    public bool AnyPinned => _slots.Any(s => s.Pins > 0);

    /// <inheritdoc/>
    public void Dispose()
    {
        if (Arena != null)
        {
            NativeMemory.AlignedFree(Arena);
            Arena = null;
        }
    }
}
