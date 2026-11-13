namespace Sharplight.Cpu;

/// <summary>Minimal parallel-for wrapper with a shared degree of parallelism.</summary>
internal static class CpuParallel
{
    private static int s_threads = Math.Max(1, Environment.ProcessorCount);
    private static ParallelOptions s_opts = new()
    {
        MaxDegreeOfParallelism = s_threads
    }
    ;

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

        if (n == 1 || Threads == 1)
        {
            for (int i = 0; i < n; i++)
            {
                body(i);
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
}
