namespace Sharplight.Cpu;

/// <summary>
/// Parallel-for with a process-wide degree of parallelism. Loops run on a dedicated spinning thread pool
/// (<see cref="CpuThreadPool"/>); a loop started while the pool is busy (another thread's loop, or a nested loop inside a
/// body) falls back to <see cref="Parallel.For(int, int, ParallelOptions, Action{int})"/>.
/// </summary>
/// <remarks>
/// Diagnostic environment switches: <c>SHARPLIGHT_CPU_THREADS</c> (initial degree of parallelism, default: logical
/// processors), <c>SHARPLIGHT_CPU_SPIN_US</c> (how long idle workers spin before blocking, default 200 µs) and
/// <c>SHARPLIGHT_CPU_POOL=0</c> (always use <see cref="Parallel"/>, for comparisons).
/// </remarks>
internal static class CpuParallel
{
    private const int DefaultSpinMicroseconds = 200;
    private static readonly bool s_usePool = Environment.GetEnvironmentVariable("SHARPLIGHT_CPU_POOL") != "0";
    private static readonly Lazy<CpuThreadPool> s_pool = new(CreatePool);
    private static int s_threads = InitialThreads();
    private static ParallelOptions s_opts = new() { MaxDegreeOfParallelism = s_threads };
    private static int s_busy;

    /// <summary>Degree of parallelism used by all kernels (process-wide).</summary>
    public static int Threads
    {
        get => s_threads;
        set
        {
            s_threads = Math.Max(1, value);
            s_opts = new ParallelOptions { MaxDegreeOfParallelism = s_threads };
        }
    }

    public static void For(int n, Action<int> body)
    {
        if (n <= 0)
        {
            return;
        }

        int threads = Threads;
        if (n == 1 || threads == 1)
        {
            for (int i = 0; i < n; i++)
            {
                body(i);
            }
            return;
        }

        if (s_usePool && Interlocked.CompareExchange(ref s_busy, 1, 0) == 0)
        {
            try
            {
                s_pool.Value.Run(n, Math.Min(n, threads) - 1, body);
            }
            finally
            {
                Volatile.Write(ref s_busy, 0);
            }

            return;
        }

        Parallel.For(0, n, s_opts, body);
    }

    /// <summary>Number of chunks for <paramref name="total"/> work items of which each chunk should have at least <paramref name="minChunk"/>.</summary>
    public static int Chunks(long total, long minChunk)
    {
        if (total <= minChunk)
        {
            return 1;
        }

        long c = Math.Min(Threads * 4L, total / minChunk);
        return (int)Math.Max(1, c);
    }

    private static int InitialThreads() =>
        int.TryParse(Environment.GetEnvironmentVariable("SHARPLIGHT_CPU_THREADS"), out var t) && t > 0 ? t : Math.Max(1, Environment.ProcessorCount);

    private static CpuThreadPool CreatePool()
    {
        int spin = int.TryParse(Environment.GetEnvironmentVariable("SHARPLIGHT_CPU_SPIN_US"), out var s) && s >= 0 ? s : DefaultSpinMicroseconds;
        return new CpuThreadPool(Math.Max(Environment.ProcessorCount, s_threads) - 1, spin);
    }
}
