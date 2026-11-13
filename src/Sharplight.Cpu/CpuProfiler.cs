using System.Collections.Concurrent;
using System.Diagnostics;
using Sharplight.Models;
using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>
/// Optional per-op wall-clock profile of the CPU provider, enabled with the diagnostic environment switch
/// <c>SHARPLIGHT_CPU_PROFILE=1</c>. Times are accumulated per op name, separately for single-token runs (decode) and
/// multi-token runs (prefill, ONNX), and printed to stderr when the process exits.
/// </summary>
internal static class CpuProfiler
{
    private const int KeyPrefixLength = 5;
    private static readonly ConcurrentDictionary<string, long[]> s_ticks = new(StringComparer.Ordinal);

    /// <summary>Whether profiling is on (read once per process).</summary>
    public static readonly bool Enabled = Environment.GetEnvironmentVariable("SHARPLIGHT_CPU_PROFILE") is "1" or "true";

    static CpuProfiler()
    {
        if (Enabled)
        {
            AppDomain.CurrentDomain.ProcessExit += (_, _) => Print();
        }
    }

    /// <summary>Runs <paramref name="op"/> and adds its elapsed time to the profile.</summary>
    public static void Run(CpuOp op, RunParameters parameters)
    {
        long start = Stopwatch.GetTimestamp();
        op.Run(parameters);
        long elapsed = Stopwatch.GetTimestamp() - start;
        bool single = parameters.Symbols.TryGetValue(WellKnown.TokenSymbol, out var tokens) && tokens == 1;
        var slot = s_ticks.GetOrAdd((single ? "T=1  " : "T>1  ") + op.Name, _ => new long[2]);
        Interlocked.Add(ref slot[0], elapsed);
        Interlocked.Increment(ref slot[1]);
    }

    private static void Print()
    {
        foreach (var group in s_ticks.GroupBy(kv => kv.Key[..KeyPrefixLength]))
        {
            double total = group.Sum(kv => kv.Value[0]) * 1000.0 / Stopwatch.Frequency;
            Console.Error.WriteLine($"cpu profile {group.Key.Trim()}: {total:F1} ms total");
            foreach (var kv in group.OrderByDescending(kv => kv.Value[0]).Take(25))
            {
                double ms = kv.Value[0] * 1000.0 / Stopwatch.Frequency;
                Console.Error.WriteLine($"  {kv.Key[KeyPrefixLength..],-40} {ms,10:F1} ms {100 * ms / total,5:F1}% {kv.Value[1],8} calls");
            }
        }
    }
}
