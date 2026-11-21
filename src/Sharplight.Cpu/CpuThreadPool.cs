using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace Sharplight.Cpu;

/// <summary>
/// Dedicated worker threads for <see cref="CpuParallel.For"/>. A decode step runs hundreds of short parallel ops, for which
/// waking thread-pool tasks costs more than the work; these workers spin for a short time after each job (like llama.cpp's
/// threadpool) so the next op starts within microseconds, and block on a monitor once idle. One job runs at a time: the
/// caller claims work items through the job's counter together with the workers and returns when all items are done.
/// Every job has its own <see cref="CpuThreadJob"/>, so a worker still leaving a finished job cannot claim items of the next.
/// </summary>
internal sealed class CpuThreadPool
{
    private const int SpinChecksPerClockRead = 256;

    private readonly object _gate = new();
    private readonly long _spinTicks;
    private readonly long _wakeTicks;
    /// <summary>Timestamp until which idle workers spin after a <see cref="Wake"/>.</summary>
    private long _wakeUntil;
    /// <summary>Workers blocked on the monitor (they spin again when a <see cref="Wake"/> pulses it).</summary>
    private int _sleepers;
    private long _wakes;
    private readonly int _workers;
    private CpuThreadJob? _job;
    private long _generation;

    /// <summary>
    /// Starts <paramref name="workers"/> background threads that spin <paramref name="spinMicroseconds"/> after each job and
    /// <paramref name="wakeMicroseconds"/> after a <see cref="Wake"/>.
    /// </summary>
    public CpuThreadPool(int workers, int spinMicroseconds, int wakeMicroseconds)
    {
        _workers = workers;
        _spinTicks = Stopwatch.Frequency * spinMicroseconds / 1_000_000;
        _wakeTicks = Stopwatch.Frequency * wakeMicroseconds / 1_000_000;
        for (int i = 0; i < workers; i++)
        {
            int id = i;
            var thread = new Thread(() => WorkerLoop(id))
            {
                IsBackground = true,
                Name = $"sharplight-cpu-{i}",
            };
            thread.Start();
        }
    }

    /// <summary>Number of worker threads (the caller is an additional participant).</summary>
    public int Workers => _workers;

    /// <summary>
    /// Runs <paramref name="body"/>(0..<paramref name="count"/>-1) on the caller and up to <paramref name="helpers"/> workers.
    /// Not reentrant: callers serialize through <see cref="CpuParallel"/>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void Run(int count, int helpers, Action<int> body)
    {
        var job = new CpuThreadJob(body, count, Math.Min(helpers, _workers));
        lock (_gate)
        {
            Volatile.Write(ref _job, job);
            _generation++;
            Monitor.PulseAll(_gate);
        }

        job.Work(0);
        var spinner = new SpinWait();
        while (!job.IsComplete)
        {
            spinner.SpinOnce(sleep1Threshold: -1);
        }

        job.Error?.Throw();
    }

    /// <summary>
    /// Has the workers spin for the wake time from now, waking the blocked ones, so that a job expected soon (the CPU part of a pass that
    /// follows a GPU segment) starts without the wake-ups: blocked workers reacquire the monitor one after another, which takes longer than
    /// the small jobs of a decode step.
    /// </summary>
    public void Wake()
    {
        // Full fences on both sides (here and in WaitForJob): a worker going to sleep sees the deadline, or this sees the sleeper.
        Interlocked.Exchange(ref _wakeUntil, Stopwatch.GetTimestamp() + _wakeTicks);
        if (Volatile.Read(ref _sleepers) == 0)
        {
            return;
        }

        lock (_gate)
        {
            _wakes++;
            Monitor.PulseAll(_gate);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void WorkerLoop(int id)
    {
        long seen = 0;
        while (true)
        {
            seen = WaitForJob(seen);
            var job = Volatile.Read(ref _job);
            if (job is not null && id < job.Participants)
            {
                job.Work(id + 1);
            }
        }
    }

    /// <summary>
    /// Spins for the configured time after the last job (or until the wake deadline), then blocks until the generation changes or a
    /// <see cref="Wake"/> restarts the spinning.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private long WaitForJob(long seen)
    {
        while (true)
        {
            long start = Stopwatch.GetTimestamp();
            int checks = 0;
            while (true)
            {
                long generation = Volatile.Read(ref _generation);
                if (generation != seen)
                {
                    return generation;
                }

                Thread.SpinWait(16);
                if (++checks % SpinChecksPerClockRead == 0)
                {
                    long now = Stopwatch.GetTimestamp();
                    if (now - start > _spinTicks && now > Volatile.Read(ref _wakeUntil))
                    {
                        break;
                    }
                }
            }

            Interlocked.Increment(ref _sleepers);
            try
            {
                if (Stopwatch.GetTimestamp() <= Interlocked.Read(ref _wakeUntil))
                {
                    continue;
                }

                lock (_gate)
                {
                    long wakes = _wakes;
                    while (_generation == seen && _wakes == wakes)
                    {
                        Monitor.Wait(_gate);
                    }

                    if (_generation != seen)
                    {
                        return _generation;
                    }
                }
            }
            finally
            {
                Interlocked.Decrement(ref _sleepers);
            }
        }
    }
}
