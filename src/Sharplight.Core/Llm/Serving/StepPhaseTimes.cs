using Sharplight.Runtime;

namespace Sharplight.Llm.Serving;

/// <summary>Host-side phases of one decoder step, in ms (diagnostics: <see cref="StepProfiler"/>).</summary>
/// <param name="PackMs">Validating the inputs and filling the host buffers (and growing KV blocks).</param>
/// <param name="UploadMs">Copying the step's inputs to the device.</param>
/// <param name="RunMs">The compiled model's run (submission and, for providers that wait, the device time).</param>
/// <param name="ReadMs">Reading the logits or sampled tokens back.</param>
/// <param name="Segments">The segments' own timings of the run.</param>
internal readonly record struct StepPhaseTimes(double PackMs, double UploadMs, double RunMs, double ReadMs, SegmentRunTimings Segments);
