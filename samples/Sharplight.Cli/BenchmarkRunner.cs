using System.Diagnostics;
using Sharplight.Cuda;
using Sharplight.Llm;
using Sharplight.Metal;
using Sharplight.Vulkan;

namespace Sharplight.Cli;

/// <summary>
/// Measures warm prefill and decode speed. Timings are taken after a warm-up that records the command buffers; best and
/// median are reported next to the mean because they are robust against other applications using the GPU.
/// </summary>
internal sealed class BenchmarkRunner
{
    private const int DecodeSteps = 64;
    private const int DefaultPrefillRepeats = 4;
    private const string PrefillRepeatsVariable = "SHARPLIGHT_BENCH_PREFILL_REPEATS";
    private const int RandomSeed = 1;
    private const int FirstSyntheticTokenId = 1000;
    private const int EndSyntheticTokenId = 50000;
    private const int WarmUpDecodeTokenId = 5;
    private const string DumpScheduleVariable = "SHARPLIGHT_DUMP_SCHEDULE";

    private readonly InferenceSession _session;
    private readonly TextWriter _log;
    private readonly int[] _promptTokens;

    public BenchmarkRunner(InferenceSession session, int promptTokenCount, TextWriter log)
    {
        _session = session;
        _log = log;
        _promptTokens = CreateSyntheticPrompt(promptTokenCount);
    }

    /// <summary>
    /// Timed prefills after the first one (<c>SHARPLIGHT_BENCH_PREFILL_REPEATS</c>, default 4): long prompts (128K) take minutes each.
    /// </summary>
    private static int PrefillRepeats
        => int.TryParse(Environment.GetEnvironmentVariable(PrefillRepeatsVariable), out int repeats) && repeats >= 0 ? repeats : DefaultPrefillRepeats;

    private IEnumerable<VulkanSegment> VulkanSegments => _session.Compiled.Segments.OfType<VulkanSegment>();

    private IEnumerable<MetalSegment> MetalSegments => _session.Compiled.Segments.OfType<MetalSegment>();

    private IEnumerable<CudaSegment> CudaSegments => _session.Compiled.Segments.OfType<CudaSegment>();

    /// <summary>Runs the benchmark and returns the one-line summary.</summary>
    public string Run(CancellationToken cancellationToken)
    {
        WarmUp();
        PrintSegmentStatistics();

        ResetProfiles();
        double firstPrefillSeconds = TimeSeconds(() => _session.Process(_promptTokens));
        PrintProfiles($"prefill T={_promptTokens.Length}");

        ResetProfiles();
        var prefillSeconds = MeasurePrefills(firstPrefillSeconds, cancellationToken);

        var decode = MeasureDecode(cancellationToken);
        PrintProfiles($"decode x{DecodeSteps}");
        if (_session.Compiled.TakeSegmentTimings() is { } timings)
        {
            _log.WriteLine($"--- decode host time per segment\n{timings}");
        }

        return FormatSummary(prefillSeconds, decode);
    }

    private static int[] CreateSyntheticPrompt(int count)
    {
        var random = new Random(RandomSeed);
        return [.. Enumerable.Range(0, count).Select(_ => random.Next(FirstSyntheticTokenId, EndSyntheticTokenId))];
    }

    private static double TimeSeconds(Action action)
    {
        var stopwatch = Stopwatch.StartNew();
        action();
        return stopwatch.Elapsed.TotalSeconds;
    }

    private void WarmUp()
    {
        _session.Process(_promptTokens);
        _session.Process([WarmUpDecodeTokenId]);
        _session.Reset();
    }

    /// <summary>The timed prefills after the warm-up (the first one and <see cref="PrefillRepeats"/> more), sorted.</summary>
    private List<double> MeasurePrefills(double firstSeconds, CancellationToken cancellationToken)
    {
        var seconds = new List<double>(PrefillRepeats + 1) { firstSeconds };
        for (int repeat = 0; repeat < PrefillRepeats; repeat++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            _session.Reset();
            seconds.Add(TimeSeconds(() => _session.Process(_promptTokens)));
        }

        seconds.Sort();
        return seconds;
    }

    private DecodeMeasurement MeasureDecode(CancellationToken cancellationToken)
    {
        _session.Reset();
        _session.Process(_promptTokens);
        ResetProfiles();
        _session.Compiled.TakeSegmentTimings();

        var stepSeconds = new List<double>(DecodeSteps);
        var gpuMilliseconds = new List<double>(DecodeSteps);
        for (int step = 0; step < DecodeSteps; step++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int token = _promptTokens[step % _promptTokens.Length];
            stepSeconds.Add(TimeSeconds(() => _session.Process([token])));
            gpuMilliseconds.Add(VulkanSegments.Sum(segment => segment.LastGpuMs) + MetalSegments.Sum(segment => segment.LastGpuMs)
                + CudaSegments.Sum(segment => segment.LastGpuMs));
        }

        return new DecodeMeasurement(stepSeconds, gpuMilliseconds);
    }

    private string FormatSummary(List<double> prefillSeconds, DecodeMeasurement decode)
    {
        double bestPrefillSeconds = prefillSeconds[0];
        double medianPrefillSeconds = Median(prefillSeconds);
        int promptTokens = _promptTokens.Length;
        var steps = decode.SortedStepSeconds;
        var gpu = decode.SortedGpuMilliseconds;
        double meanTokensPerSecond = DecodeSteps / decode.TotalSeconds;
        return $"prefill {promptTokens} tok: best {bestPrefillSeconds * 1000:F0} ms = {promptTokens / bestPrefillSeconds:F1} tok/s"
            + $" | decode: best {1 / steps[0]:F2}, median {1 / Median(steps):F2}, mean {meanTokensPerSecond:F2} tok/s"
            + $" | GPU ms/step: best {gpu[0]:F2}, median {Median(gpu):F2}, wall best {steps[0] * 1000:F2}"
            + $" | prefill median of {prefillSeconds.Count}: {medianPrefillSeconds * 1000:F1} ms = {promptTokens / medianPrefillSeconds:F1} tok/s";
    }

    private static double Median(IReadOnlyList<double> sorted)
    {
        return sorted[sorted.Count / 2];
    }

    private void ResetProfiles()
    {
        foreach (var segment in VulkanSegments)
        {
            segment.ResetProfile();
        }

        foreach (var segment in MetalSegments)
        {
            segment.ResetProfile();
        }

        foreach (var segment in CudaSegments)
        {
            segment.ResetProfile();
        }
    }

    private void PrintProfiles(string phase)
    {
        foreach (var segment in VulkanSegments.Where(segment => segment.Profile))
        {
            _log.WriteLine($"--- {phase} profile, segment {segment.Segment.Index}");
            _log.WriteLine(segment.ProfileReport());
        }

        foreach (var segment in MetalSegments.Where(segment => segment.Profile))
        {
            _log.WriteLine($"--- {phase} profile, segment {segment.Segment.Index}");
            _log.WriteLine(segment.ProfileReport());
        }

        foreach (var segment in CudaSegments.Where(segment => segment.Profile))
        {
            _log.WriteLine($"--- {phase} profile, segment {segment.Segment.Index}");
            _log.WriteLine(segment.ProfileReport());
        }
    }

    private void PrintSegmentStatistics()
    {
        bool dumpSchedule = Environment.GetEnvironmentVariable(DumpScheduleVariable) == "1";
        foreach (var segment in VulkanSegments)
        {
            var (dispatches, barriers) = segment.Stats(1);
            _log.WriteLine($"segment {segment.Segment.Index}: {dispatches} dispatches, {barriers} barriers per step");
            if (dumpSchedule)
            {
                PrintSchedule(segment);
            }
        }

        foreach (var segment in MetalSegments)
        {
            var (dispatches, barriers) = segment.Stats(1);
            _log.WriteLine($"segment {segment.Segment.Index}: {dispatches} dispatches, {barriers} barriers per step (metal)");
        }

        foreach (var segment in CudaSegments)
        {
            var (launches, _) = segment.Stats(1);
            _log.WriteLine($"segment {segment.Segment.Index}: {launches} launches per step (cuda, one stream)");
        }
    }

    private void PrintSchedule(VulkanSegment segment)
    {
        foreach (var (name, barrierBefore) in segment.Schedule())
        {
            if (barrierBefore)
            {
                _log.WriteLine("--- barrier");
            }

            _log.WriteLine($"    {name}");
        }
    }

    private sealed class DecodeMeasurement
    {
        public DecodeMeasurement(List<double> stepSeconds, List<double> gpuMilliseconds)
        {
            TotalSeconds = stepSeconds.Sum();
            stepSeconds.Sort();
            gpuMilliseconds.Sort();
            SortedStepSeconds = stepSeconds;
            SortedGpuMilliseconds = gpuMilliseconds;
        }

        public double TotalSeconds { get; }

        public IReadOnlyList<double> SortedStepSeconds { get; }

        public IReadOnlyList<double> SortedGpuMilliseconds { get; }
    }
}
