using System.Diagnostics;

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
    private readonly int _workers;
    private CpuThreadJob? _job;
    private long _generation;

    /// <summary>Starts <paramref name="workers"/> background threads that spin <paramref name="spinMicroseconds"/> after each job.</summary>
    public CpuThreadPool(int workers, int spinMicroseconds)
    {
        _workers = workers;
        _spinTicks = Stopwatch.Frequency * spinMicroseconds / 1_000_000;
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
    public void Run(int count, int helpers, Action<int> body)
    {
        var job = new CpuThreadJob(body, count, Math.Min(helpers, _workers));
        lock (_gate)
        {
            Volatile.Write(ref _job, job);
            _generation++;
            Monitor.PulseAll(_gate);
        }

        job.Work();
        var spinner = new SpinWait();
        while (!job.IsComplete)
        {
            spinner.SpinOnce(sleep1Threshold: -1);
        }

        job.Error?.Throw();
    }

    private void WorkerLoop(int id)
    {
        long seen = 0;
        while (true)
        {
            seen = WaitForJob(seen);
            var job = Volatile.Read(ref _job);
            if (job is not null && id < job.Participants)
            {
                job.Work();
            }
        }
    }

    /// <summary>Spins for the configured time after the last job, then blocks until the generation changes.</summary>
    private long WaitForJob(long seen)
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
            if (++checks % SpinChecksPerClockRead == 0 && Stopwatch.GetTimestamp() - start > _spinTicks)
            {
                break;
            }
        }

        lock (_gate)
        {
            while (_generation == seen)
            {
                Monitor.Wait(_gate);
            }

            return _generation;
        }
    }
}
