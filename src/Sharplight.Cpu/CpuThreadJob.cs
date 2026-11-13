using System.Runtime.ExceptionServices;

namespace Sharplight.Cpu;

/// <summary>One parallel loop of <see cref="CpuThreadPool"/>: items are claimed through an atomic counter.</summary>
internal sealed class CpuThreadJob(Action<int> body, int count, int participants)
{
    private int _next;
    private int _done;
    private ExceptionDispatchInfo? _error;

    /// <summary>Workers with an id below this value help with the job.</summary>
    public int Participants { get; } = participants;

    /// <summary>All items have finished.</summary>
    public bool IsComplete => Volatile.Read(ref _done) >= count;

    /// <summary>The first exception thrown by an item, if any.</summary>
    public ExceptionDispatchInfo? Error => Volatile.Read(ref _error);

    /// <summary>Claims and runs items until none are left.</summary>
    public void Work()
    {
        int i;
        while ((i = Interlocked.Increment(ref _next) - 1) < count)
        {
            try
            {
                body(i);
            }
            catch (Exception e)
            {
                Interlocked.CompareExchange(ref _error, ExceptionDispatchInfo.Capture(e), null);
            }

            Interlocked.Increment(ref _done);
        }
    }
}
