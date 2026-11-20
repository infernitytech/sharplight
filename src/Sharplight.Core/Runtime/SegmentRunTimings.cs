namespace Sharplight.Runtime;

/// <summary>
/// Where the most recent execution of a compiled segment spent its time (diagnostics: serving step profiles). Providers that do
/// not measure report zeros.
/// </summary>
/// <param name="GpuMs">Device time from the first to the last launch, in ms.</param>
/// <param name="RecordMs">Host time spent resolving the launches of a shape run for the first time (or after eviction), in ms.</param>
/// <param name="SubmitMs">Host time spent submitting the work (launch calls, graph launch, input copies of the execution), in ms.</param>
/// <param name="WaitMs">Host time spent waiting for the device after submitting, in ms.</param>
/// <param name="Recorded">True when this execution recorded its shape (it had not run before or was evicted).</param>
/// <param name="Replayed">True when the work was replayed as one recorded unit (e.g. a CUDA graph) rather than launch by launch.</param>
public readonly record struct SegmentRunTimings(double GpuMs, double RecordMs, double SubmitMs, double WaitMs, bool Recorded, bool Replayed)
{
    /// <summary>Adds the timings of two segments of one run.</summary>
    /// <param name="a">The first segment's timings.</param>
    /// <param name="b">The second segment's timings.</param>
    /// <returns>The summed times; flags are true when either segment set them.</returns>
    public static SegmentRunTimings operator +(SegmentRunTimings a, SegmentRunTimings b)
        => new(a.GpuMs + b.GpuMs, a.RecordMs + b.RecordMs, a.SubmitMs + b.SubmitMs, a.WaitMs + b.WaitMs, a.Recorded || b.Recorded, a.Replayed || b.Replayed);
}
