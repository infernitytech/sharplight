using Sharplight.Tensors;

namespace Sharplight.Llm;

/// <summary>
/// Every <c>SHARPLIGHT_*</c> environment switch read by the Llm project, read once per process. The defaults are the shipped behavior.
/// <list type="bullet">
/// <item><c>SHARPLIGHT_DEVICE_SAMPLING=0</c>: never sample tokens on the accelerator; logits are downloaded and sampled on the CPU
/// (same as <see cref="LlmLoadOptions.DeviceSampling"/> = false).</item>
/// <item><c>SHARPLIGHT_RECURRENT_STATE=f32|bf16|f16</c>: element type of 16-bit-capable recurrent states, overriding
/// <see cref="LlmLoadOptions.RecurrentStateType"/> and the bf16 default of <see cref="Tasks.TaskLoadOptions.AllowReducedPrecision"/>
/// (e.g. f32 isolates the other reduced-precision paths, bf16 without reduced precision isolates the state).</item>
/// <item><c>SHARPLIGHT_SCHED_TRACE=1</c>: the serving scheduler writes one line per step to stderr (duration, rows, decode rows, prompt
/// chunks with the reason they end, prefix-cache attaches and state snapshots).</item>
/// <item><c>SHARPLIGHT_STEP_SNAPSHOTS=0</c>: recurrent-state snapshots for prefix caching are only copied after a step, so a snapshot
/// inside a prompt chunk cuts the chunk (<see cref="Serving.IBatchDecoder.SnapshotsInsideSteps"/> is false; for A/B measurements).</item>
/// </list>
/// </summary>
internal static class DiagnosticSwitches
{
    /// <summary>False when device sampling is disabled for the process.</summary>
    public static bool DeviceSampling { get; } = Environment.GetEnvironmentVariable("SHARPLIGHT_DEVICE_SAMPLING") != "0";

    /// <summary>True when the serving scheduler traces its steps.</summary>
    public static bool SchedulerTrace { get; } = Environment.GetEnvironmentVariable("SHARPLIGHT_SCHED_TRACE") == "1";

    /// <summary>False when steps must not keep recurrent state inside an input (<c>SHARPLIGHT_STEP_SNAPSHOTS=0</c>).</summary>
    public static bool StepSnapshots { get; } = Environment.GetEnvironmentVariable("SHARPLIGHT_STEP_SNAPSHOTS") != "0";

    /// <summary>The recurrent state type forced for the process, or null.</summary>
    public static DataType? RecurrentState { get; } = Environment.GetEnvironmentVariable("SHARPLIGHT_RECURRENT_STATE")?.ToLowerInvariant() switch
    {
        null or "" => null,
        "f32" => DataType.F32,
        "bf16" => DataType.BF16,
        "f16" => DataType.F16,
        var other => throw new InvalidOperationException($"SHARPLIGHT_RECURRENT_STATE must be f32, bf16 or f16, got '{other}'."),
    };
}
