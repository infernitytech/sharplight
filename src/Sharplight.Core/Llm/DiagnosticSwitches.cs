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
/// <item><c>SHARPLIGHT_STEP_PROFILE=&lt;file&gt;</c>: the serving scheduler writes one CSV line per step to the file (rows, scheduler phases,
/// device time; see <see cref="Serving.StepProfiler"/>).</item>
/// <item><c>SHARPLIGHT_ASYNC_EMIT=0</c>: the serving scheduler decodes and streams each step's tokens before it schedules the next step,
/// also when the decoder runs steps asynchronously (by default that work overlaps the next step on the device).</item>
/// <item><c>SHARPLIGHT_STEP_SNAPSHOTS=0</c>: recurrent-state snapshots for prefix caching are only copied after a step, so a snapshot
/// inside a prompt chunk cuts the chunk (<see cref="Serving.IBatchDecoder.SnapshotsInsideSteps"/> is false; for A/B measurements).</item>
/// <item><c>SHARPLIGHT_DEVICE_RESERVE_MIB=&lt;n&gt;</c>: device memory kept free on every device by a placement that
/// <see cref="LlmLoadOptions.ReservePerDevice"/> leaves at its default (for measurements).</item>
/// </list>
/// </summary>
internal static class DiagnosticSwitches
{
    /// <summary>The device reserve forced for the process in bytes, or null (<c>SHARPLIGHT_DEVICE_RESERVE_MIB</c>).</summary>
    public static long? DeviceReserve { get; } = long.TryParse(Environment.GetEnvironmentVariable("SHARPLIGHT_DEVICE_RESERVE_MIB"), out long mib) && mib >= 0 ? mib << 20 : null;

    /// <summary>False when device sampling is disabled for the process.</summary>
    public static bool DeviceSampling { get; } = Environment.GetEnvironmentVariable("SHARPLIGHT_DEVICE_SAMPLING") != "0";

    /// <summary>True when the serving scheduler traces its steps.</summary>
    public static bool SchedulerTrace { get; } = Environment.GetEnvironmentVariable("SHARPLIGHT_SCHED_TRACE") == "1";

    /// <summary>False when token emission must not overlap the next step (<c>SHARPLIGHT_ASYNC_EMIT=0</c>).</summary>
    public static bool DeferEmits { get; } = Environment.GetEnvironmentVariable("SHARPLIGHT_ASYNC_EMIT") != "0";

    /// <summary>
    /// Prototype policies of the load-adaptive step budget for prompt waves (<c>SHARPLIGHT_STEP_FLOOR_WAVES</c>, benchmarks.md section 93;
    /// off by default): <c>1</c> applies the floor (<see cref="Serving.EngineOptions.MinBatchTokens"/>) only while no sequence decodes,
    /// <c>full</c> gives every step the whole budget while some sequence decodes. Null: the default policy.
    /// </summary>
    public static string? StepFloorWaves { get; } = Environment.GetEnvironmentVariable("SHARPLIGHT_STEP_FLOOR_WAVES") switch
    {
        "1" => "1",
        "full" => "full",
        _ => null,
    };

    /// <summary>The file of the serving step profile, or null.</summary>
    public static string? StepProfile { get; } = Environment.GetEnvironmentVariable("SHARPLIGHT_STEP_PROFILE");

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
