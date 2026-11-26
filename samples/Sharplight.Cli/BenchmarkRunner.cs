using System.Diagnostics;
using Sharplight.Cuda;
using Sharplight.Llm;
using Sharplight.Metal;
using Sharplight.Runtime.Experts;
using Sharplight.Vulkan;

namespace Sharplight.Cli;

/// <summary>
/// Measures warm prefill and decode speed. Timings are taken after a warm-up that records the command buffers; best and
/// median are reported next to the mean because they are robust against other applications using the GPU. The summary ends with
/// llama-bench's statistic: the mean and sample standard deviation of the per-run rates (prefill: prompt tokens over one timed
/// prefill from an empty cache; decode: 64 tokens over one run of 64 steps after the prompt, <c>llama-bench -n 64 -d N</c>).
/// </summary>
internal sealed class BenchmarkRunner
{
    private const int DecodeSteps = 64;
    private const int DefaultPrefillRepeats = 4;
    private const string PrefillRepeatsVariable = "SHARPLIGHT_BENCH_PREFILL_REPEATS";
    private const int DefaultDecodeRuns = 1;
    private const string DecodeRunsVariable = "SHARPLIGHT_BENCH_DECODE_RUNS";
    private const int RandomSeed = 1;
    private const int FirstSyntheticTokenId = 1000;
    private const int EndSyntheticTokenId = 50000;
    private const int WarmUpDecodeTokenId = 5;
    private const string DumpScheduleVariable = "SHARPLIGHT_DUMP_SCHEDULE";
    private const string ProfileTopVariable = "SHARPLIGHT_PROFILE_TOP";
    private const string PromptVariable = "SHARPLIGHT_BENCH_PROMPT";
    private const string WholeVocabularyPrompt = "vocab";

    private readonly InferenceSession _session;
    private readonly TextWriter _log;
    private readonly int[] _promptTokens;
    private readonly ExpertStore? _diskExperts;

    public BenchmarkRunner(InferenceSession session, int promptTokenCount, TextWriter log, ExpertStore? diskExperts = null)
    {
        _session = session;
        _log = log;
        _diskExperts = diskExperts is { IsActive: true } ? diskExperts : null;
        _promptTokens = CreateSyntheticPrompt(promptTokenCount, session.VocabSize);
    }

    /// <summary>
    /// Timed prefills after the first one (<c>SHARPLIGHT_BENCH_PREFILL_REPEATS</c>, default 4): long prompts (128K) take minutes each.
    /// </summary>
    private static int PrefillRepeats
        => int.TryParse(Environment.GetEnvironmentVariable(PrefillRepeatsVariable), out int repeats) && repeats >= 0 ? repeats : DefaultPrefillRepeats;

    /// <summary>
    /// Timed decode runs of <see cref="DecodeSteps"/> steps (<c>SHARPLIGHT_BENCH_DECODE_RUNS</c>, default 1); each run resets the
    /// session and prefills the prompt untimed first, as llama-bench's repetitions of a tg test at a depth do.
    /// </summary>
    private static int DecodeRuns
        => int.TryParse(Environment.GetEnvironmentVariable(DecodeRunsVariable), out int runs) && runs >= 1 ? runs : DefaultDecodeRuns;

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

        if (_diskExperts is not null)
        {
            _log.WriteLine($"experts on disk, warm-up and prefills: {_diskExperts.Statistics}");
            _diskExperts.ResetStatistics();
        }

        var decode = MeasureDecode(cancellationToken);
        PrintProfiles($"decode x{DecodeSteps}");
        if (_diskExperts is not null)
        {
            _log.WriteLine($"experts on disk, decode: {_diskExperts.Statistics}");
            _diskExperts.ResetStatistics();
        }
        if (_session.Compiled.TakeSegmentTimings() is { } timings)
        {
            _log.WriteLine($"--- decode host time per segment\n{timings}");
        }

        return FormatSummary(prefillSeconds, decode);
    }

    /// <summary>
    /// Random token ids from [<see cref="FirstSyntheticTokenId"/>, <see cref="EndSyntheticTokenId"/>), or with <c>SHARPLIGHT_BENCH_PROMPT=vocab</c>
    /// from the whole vocabulary, as llama-bench draws its prompts (the experts a mixture-of-experts prompt selects depend on its tokens).
    /// </summary>
    private static int[] CreateSyntheticPrompt(int count, int vocabularySize)
    {
        var random = new Random(RandomSeed);
        bool wholeVocabulary = Environment.GetEnvironmentVariable(PromptVariable) == WholeVocabularyPrompt;
        int first = wholeVocabulary ? 0 : FirstSyntheticTokenId;
        int end = wholeVocabulary ? vocabularySize : EndSyntheticTokenId;
        return [.. Enumerable.Range(0, count).Select(_ => random.Next(first, end))];
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
        int runs = DecodeRuns;
        var stepSeconds = new List<double>(DecodeSteps * runs);
        var gpuMilliseconds = new List<double>(DecodeSteps * runs);
        var runSeconds = new List<double>(runs);
        for (int run = 0; run < runs; run++)
        {
            _session.Reset();
            _session.Process(_promptTokens);
            if (run == 0)
            {
                ResetProfiles();
                _session.Compiled.TakeSegmentTimings();
            }

            double seconds = 0;
            for (int step = 0; step < DecodeSteps; step++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                int token = _promptTokens[step % _promptTokens.Length];
                double stepTime = TimeSeconds(() => _session.Process([token]));
                seconds += stepTime;
                stepSeconds.Add(stepTime);
                gpuMilliseconds.Add(VulkanSegments.Sum(segment => segment.LastGpuMs) + MetalSegments.Sum(segment => segment.LastGpuMs)
                    + CudaSegments.Sum(segment => segment.LastGpuMs));
            }

            runSeconds.Add(seconds);
        }

        return new DecodeMeasurement(stepSeconds, gpuMilliseconds, runSeconds);
    }

    private string FormatSummary(List<double> prefillSeconds, DecodeMeasurement decode)
    {
        double bestPrefillSeconds = prefillSeconds[0];
        double medianPrefillSeconds = Median(prefillSeconds);
        int promptTokens = _promptTokens.Length;
        var steps = decode.SortedStepSeconds;
        var gpu = decode.SortedGpuMilliseconds;
        double meanTokensPerSecond = steps.Count / decode.TotalSeconds;
        var prefillRates = prefillSeconds.Select(seconds => promptTokens / seconds).ToList();
        var decodeRates = decode.RunSeconds.Select(seconds => DecodeSteps / seconds).ToList();
        return $"prefill {promptTokens} tok: best {bestPrefillSeconds * 1000:F0} ms = {promptTokens / bestPrefillSeconds:F1} tok/s"
            + $" | decode: best {1 / steps[0]:F2}, median {1 / Median(steps):F2}, mean {meanTokensPerSecond:F2} tok/s"
            + $" | GPU ms/step: best {gpu[0]:F2}, median {Median(gpu):F2}, wall best {steps[0] * 1000:F2}"
            + $" | prefill median of {prefillSeconds.Count}: {medianPrefillSeconds * 1000:F1} ms = {promptTokens / medianPrefillSeconds:F1} tok/s"
            + $" | per run: prefill {prefillRates.Average():F1} ± {StandardDeviation(prefillRates):F1} tok/s of {prefillRates.Count},"
            + $" decode {decodeRates.Average():F2} ± {StandardDeviation(decodeRates):F2} tok/s of {decodeRates.Count}"
            + $" ({string.Join(", ", decodeRates.Select(rate => rate.ToString("F2")))})";
    }

    private static double Median(IReadOnlyList<double> sorted)
    {
        return sorted[sorted.Count / 2];
    }

    /// <summary>Sample standard deviation (n - 1), as llama-bench reports it; 0 for fewer than two values.</summary>
    private static double StandardDeviation(IReadOnlyList<double> values)
    {
        if (values.Count < 2)
        {
            return 0;
        }

        double mean = values.Average();
        return Math.Sqrt(values.Sum(value => (value - mean) * (value - mean)) / (values.Count - 1));
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
        // SHARPLIGHT_PROFILE_TOP=n prints the n heaviest entries instead of 30 (the Vulkan entries are per node).
        int top = int.TryParse(Environment.GetEnvironmentVariable(ProfileTopVariable), out int n) && n > 0 ? n : 30;
        foreach (var segment in VulkanSegments.Where(segment => segment.Profile))
        {
            _log.WriteLine($"--- {phase} profile, segment {segment.Segment.Index}");
            _log.WriteLine(segment.ProfileReport(top));
        }

        foreach (var segment in MetalSegments.Where(segment => segment.Profile))
        {
            _log.WriteLine($"--- {phase} profile, segment {segment.Segment.Index}");
            _log.WriteLine(segment.ProfileReport(top));
        }

        foreach (var segment in CudaSegments.Where(segment => segment.Profile))
        {
            _log.WriteLine($"--- {phase} profile, segment {segment.Segment.Index}");
            _log.WriteLine(segment.ProfileReport(top));
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
        public DecodeMeasurement(List<double> stepSeconds, List<double> gpuMilliseconds, IReadOnlyList<double> runSeconds)
        {
            TotalSeconds = stepSeconds.Sum();
            RunSeconds = runSeconds;
            stepSeconds.Sort();
            gpuMilliseconds.Sort();
            SortedStepSeconds = stepSeconds;
            SortedGpuMilliseconds = gpuMilliseconds;
        }

        public double TotalSeconds { get; }

        public IReadOnlyList<double> RunSeconds { get; }

        public IReadOnlyList<double> SortedStepSeconds { get; }

        public IReadOnlyList<double> SortedGpuMilliseconds { get; }
    }
}
