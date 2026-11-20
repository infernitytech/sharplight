using System.Diagnostics;
using System.Globalization;

namespace Sharplight.Llm.Serving;

/// <summary>
/// Writes one CSV line per serving step (<c>SHARPLIGHT_STEP_PROFILE=&lt;file&gt;</c>): when it started, its rows, and where the scheduler
/// thread's time went between this step's start and the next one's (admission, scheduling, the decoder's phases, the device time, result
/// processing). The gap between consecutive starts minus the device time is the time the device waited for the host.
/// </summary>
internal sealed class StepProfiler : IDisposable
{
    private const string Header = "start_ms,decode_rows,prompt_rows,prompt_inputs,active,admit_ms,schedule_ms,pack_ms,upload_ms,run_ms,gpu_ms,record_ms,submit_ms,wait_ms,recorded,replayed,overlap_ms,read_ms,process_ms,gc_pause_ms,gc0,gc1,gc2";
    private readonly StreamWriter _out;
    private readonly long _origin = Stopwatch.GetTimestamp();
    private int _lines;
    private TimeSpan _gcPause = GC.GetTotalPauseDuration();
    private int _gc0 = GC.CollectionCount(0);
    private int _gc1 = GC.CollectionCount(1);
    private int _gc2 = GC.CollectionCount(2);

    private StepProfiler(string path)
    {
        _out = new StreamWriter(path, append: false) { AutoFlush = false };
        _out.WriteLine(Header);
    }

    /// <summary>The profiler of this process's switch, or null when it is off.</summary>
    public static StepProfiler? Create() => DiagnosticSwitches.StepProfile is { Length: > 0 } path ? new StepProfiler(path) : null;

    /// <summary>Scheduler-thread time spent reaping and admitting before the next step, in ms.</summary>
    public double AdmitMs { get; set; }

    /// <summary>Writes one step.</summary>
    public void Write(long start, int decodeRows, int promptRows, int promptInputs, int active, double scheduleMs, StepPhaseTimes phases, double processMs)
    {
        var c = CultureInfo.InvariantCulture;
        var t = phases.Segments;
        _out.WriteLine(string.Create(c, $"{Stopwatch.GetElapsedTime(_origin, start).TotalMilliseconds:F3},{decodeRows},{promptRows},{promptInputs},{active},{AdmitMs:F3},{scheduleMs:F3},{phases.PackMs:F3},{phases.UploadMs:F3},{phases.RunMs:F3},{t.GpuMs:F3},{t.RecordMs:F3},{t.SubmitMs:F3},{t.WaitMs:F3},{(t.Recorded ? 1 : 0)},{(t.Replayed ? 1 : 0)},{phases.OverlapMs:F3},{phases.ReadMs:F3},{processMs:F3},{GcSince():F3},{Collections(0, ref _gc0)},{Collections(1, ref _gc1)},{Collections(2, ref _gc2)}"));
        AdmitMs = 0;
        if (++_lines % 256 == 0)
        {
            _out.Flush();
        }
    }

    /// <summary>Time the process spent paused in garbage collections since the previous step, in ms.</summary>
    private double GcSince()
    {
        var now = GC.GetTotalPauseDuration();
        double ms = (now - _gcPause).TotalMilliseconds;
        _gcPause = now;
        return ms;
    }

    /// <summary>Collections of generation <paramref name="generation"/> since the previous step.</summary>
    private static int Collections(int generation, ref int seen)
    {
        int now = GC.CollectionCount(generation);
        int n = now - seen;
        seen = now;
        return n;
    }

    /// <summary>Flushes and closes the file.</summary>
    public void Dispose() => _out.Dispose();
}
