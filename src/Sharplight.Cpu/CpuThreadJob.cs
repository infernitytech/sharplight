using System.Runtime.CompilerServices;
using System.Runtime.ExceptionServices;

namespace Sharplight.Cpu;

/// <summary>
/// One parallel loop of <see cref="CpuThreadPool"/>. The items are split into one contiguous range per participant
/// (the caller is participant 0, worker <c>i</c> is participant <c>i + 1</c>); each participant first works through its own
/// range, then steals from the others' ranges. Consecutive ops that split their outputs alike thus run the same region on
/// the same core, which finds the previous op's output for that region in its own caches, while a participant that is
/// descheduled (a shared machine) still has its remaining items taken by the others. Items are claimed several at a
/// time when there are many per participant; each range's counter has its own cache line.
/// </summary>
internal sealed class CpuThreadJob
{
    /// <summary>Claims per participant aimed for; fewer items per claim keep coarse loops balanced.</summary>
    private const int ClaimsPerParticipant = 8;

    /// <summary>Longs between two counters (128 bytes: adjacent-line prefetch pairs).</summary>
    private const int CounterStride = 16;

    /// <summary><c>SHARPLIGHT_CPU_AFFINITY=0</c> claims all items from one shared counter (the scheduling before ranges).</summary>
    private readonly Action<int> _body;
    private readonly int _count;
    private readonly int _ranges;
    private readonly int _batch;
    private readonly long[] _counters; // [(r + 1) * stride]: next item of range r; [(ranges + 1) * stride]: done items
    private ExceptionDispatchInfo? _error;

    public CpuThreadJob(Action<int> body, int count, int participants)
    {
        _body = body;
        _count = count;
        Participants = participants;
        _ranges = DiagnosticSwitches.SharedRanges ? 1 : Math.Max(1, Math.Min(count, participants + 1));
        _batch = Math.Max(1, count / ((participants + 1) * ClaimsPerParticipant));
        _counters = new long[(_ranges + 2) * CounterStride];
        for (int r = 0; r < _ranges; r++)
        {
            _counters[(r + 1) * CounterStride] = RangeStart(r);
        }
    }

    /// <summary>Workers with an id below this value help with the job.</summary>
    public int Participants { get; }

    /// <summary>All items have finished.</summary>
    public bool IsComplete => Volatile.Read(ref _counters[(_ranges + 1) * CounterStride]) >= _count;

    /// <summary>The first exception thrown by an item, if any.</summary>
    public ExceptionDispatchInfo? Error => Volatile.Read(ref _error);

    /// <summary>Runs items of participant <paramref name="participant"/>'s own range, then of the other ranges, until none are left.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void Work(int participant)
    {
        int own = participant % _ranges;
        for (int k = 0; k < _ranges; k++)
        {
            int r = (own + k) % _ranges;
            long end = RangeStart(r + 1);
            ref long next = ref _counters[(r + 1) * CounterStride];
            long start;
            while (Volatile.Read(ref next) < end && (start = Interlocked.Add(ref next, _batch) - _batch) < end)
            {
                long stop = Math.Min(end, start + _batch);
                for (long i = start; i < stop; i++)
                {
                    try
                    {
                        _body((int)i);
                    }
                    catch (Exception e)
                    {
                        Interlocked.CompareExchange(ref _error, ExceptionDispatchInfo.Capture(e), null);
                    }
                }

                Interlocked.Add(ref _counters[(_ranges + 1) * CounterStride], stop - start);
            }
        }
    }

    /// <summary>First item of range <paramref name="r"/> (ranges differ in size by at most one item).</summary>
    private long RangeStart(int r) => (long)_count * r / _ranges;
}
