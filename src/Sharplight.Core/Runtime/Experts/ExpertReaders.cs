namespace Sharplight.Runtime.Experts;

/// <summary>
/// The expert readers of this platform: on Linux io_uring where the kernel allows it (detected at run time: old kernels and seccomp profiles
/// of containers refuse it), else positional reads on a few threads; elsewhere the threads. <c>SHARPLIGHT_EXPERT_READER</c> (<c>io_uring</c>,
/// <c>threads</c>) picks one, <c>SHARPLIGHT_EXPERT_READ_DEPTH</c> sets the reads in flight (default 32 for io_uring, 8 threads).
/// </summary>
public static class ExpertReaders
{
    private const int DefaultRingDepth = 32;
    private const int DefaultThreads = 8;

    /// <summary>Creates the reader to use.</summary>
    /// <param name="log">Receives why io_uring is not used, or null.</param>
    public static IExpertReader CreateDefault(Action<string>? log = null)
    {
        string? choice = Environment.GetEnvironmentVariable("SHARPLIGHT_EXPERT_READER");
        int? depth = int.TryParse(Environment.GetEnvironmentVariable("SHARPLIGHT_EXPERT_READ_DEPTH"), out int d) && d > 0 ? d : null;
        if (choice is not "threads" && OperatingSystem.IsLinux())
        {
            if (IoUringExpertReader.TryCreate(depth ?? DefaultRingDepth, out var ring, out string? reason))
            {
                return ring!;
            }

            log?.Invoke($"Experts on disk: io_uring is unavailable ({reason}); reading with {depth ?? DefaultThreads} threads instead.");
        }

        return new ThreadPoolExpertReader(depth ?? DefaultThreads);
    }

    /// <summary>Creates the io_uring reader, or returns null where io_uring is unavailable.</summary>
    /// <param name="depth">Reads in flight at most.</param>
    /// <param name="reason">Why it is unavailable, or null.</param>
    public static IExpertReader? TryCreateIoUring(int depth, out string? reason)
        => IoUringExpertReader.TryCreate(depth, out var ring, out reason) ? ring : null;

    /// <summary>Creates the portable reader: positional reads on <paramref name="threads"/> threads.</summary>
    /// <param name="threads">Reads in flight at most.</param>
    public static IExpertReader CreateThreads(int threads) => new ThreadPoolExpertReader(threads);
}
