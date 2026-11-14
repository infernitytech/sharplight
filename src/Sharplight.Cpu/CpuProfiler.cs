using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using Sharplight.Models;
using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>
/// Optional per-op wall-clock profile of the CPU provider, enabled with the diagnostic environment switch
/// <c>SHARPLIGHT_CPU_PROFILE=1</c>. Times are accumulated per op name, separately for single-token runs (decode) and
/// multi-token runs (prefill, ONNX), and printed to stderr when the process exits. <c>SHARPLIGHT_CPU_PROFILE=min</c> instead
/// keeps the minimum time of every op instance over all runs and prints their sums: on a shared machine, that approximates
/// an undisturbed run better than totals.
/// </summary>
internal static class CpuProfiler
{
    private const int KeyPrefixLength = 5;
    private static readonly ConcurrentDictionary<string, long[]> s_ticks = new(StringComparer.Ordinal);

    /// <summary>Whether profiling is on (read once per process).</summary>
    public static readonly bool Enabled = Environment.GetEnvironmentVariable("SHARPLIGHT_CPU_PROFILE") is "1" or "true" or "min";

    private static readonly bool s_minimum = Environment.GetEnvironmentVariable("SHARPLIGHT_CPU_PROFILE") == "min";
    private static readonly ConcurrentDictionary<CpuOp, long> s_minTicks = new(ReferenceEqualityComparer.Instance);

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
        if (s_minimum)
        {
            s_minTicks.AddOrUpdate(op, elapsed, (_, old) => Math.Min(old, elapsed));
            return;
        }

        bool single = parameters.Symbols.TryGetValue(WellKnown.TokenSymbol, out var tokens) && tokens == 1;
        var slot = s_ticks.GetOrAdd((single ? "T=1  " : "T>1  ") + op.Name, _ => new long[2]);
        Interlocked.Add(ref slot[0], elapsed);
        Interlocked.Increment(ref slot[1]);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Print()
    {
        if (s_minimum)
        {
            PrintMinimum();
            return;
        }

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

    /// <summary>Sum of the per-instance minima, grouped by op name, then the slowest instances with their details.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void PrintMinimum()
    {
        double Ms(long ticks) => ticks * 1000.0 / Stopwatch.Frequency;
        double total = Ms(s_minTicks.Values.Sum());
        Console.Error.WriteLine($"cpu profile (sum of per-op minima): {total:F3} ms");
        foreach (var group in s_minTicks.GroupBy(kv => kv.Key.Name).OrderByDescending(g => g.Sum(kv => kv.Value)).Take(25))
        {
            double ms = Ms(group.Sum(kv => kv.Value));
            Console.Error.WriteLine($"  {group.Key,-40} {ms,9:F3} ms {100 * ms / total,5:F1}% {group.Count(),6} ops");
        }

        Console.Error.WriteLine("slowest instances:");
        foreach (var kv in s_minTicks.OrderByDescending(kv => kv.Value).Take(60))
        {
            Console.Error.WriteLine($"  {kv.Key.Name,-28} {kv.Key.Detail,-36} {Ms(kv.Value) * 1000,9:F1} us");
        }
    }
}
