using System.Text.Json.Serialization;

namespace Sharplight.Llm.Serving;

/// <summary>
/// When the floor of the load-adaptive step budget (<see cref="EngineOptions.MinBatchTokens"/>) applies (<see cref="EngineOptions.StepFloor"/>).
/// In JSON (hosting files) the names, case-insensitive: <c>waves</c>, <c>always</c>, <c>full</c>. Only steps that carry prompt rows while few sequences hold a slot differ: under load the adaptive budget is the whole
/// <see cref="EngineOptions.MaxBatchTokens"/> anyway.
/// </summary>
[JsonConverter(typeof(JsonStringEnumConverter<StepFloorPolicy>))]
public enum StepFloorPolicy
{
    /// <summary>
    /// The default: the floor applies only while no sequence decodes (a wave of prompts arriving at an idle engine). While some
    /// sequence decodes, the budget is <see cref="EngineOptions.BatchTokensPerSequence"/> rows per sequence holding a slot rounded up to
    /// <see cref="EngineOptions.BatchTokensGranule"/>, so the prompts a wave admitted first decode through shorter steps (benchmarks.md
    /// §93.4: at 16 closed-loop streams TTFT p50 -22 to -28 % and ITL p99 -20 to -28 % against <see cref="Always"/>; throughput unchanged
    /// for a dense FP8 model, -1.7 % for a mixture-of-experts model whose prompt steps read every expert).
    /// </summary>
    Waves,

    /// <summary>The floor applies to every step (the behavior before 2026-11-23).</summary>
    Always,

    /// <summary>
    /// While some sequence decodes every step may take the whole budget; the floor applies only while no sequence decodes. The lowest
    /// ITL tail and the highest throughput of the three, the longest time to first token in prompt waves.
    /// </summary>
    Full,
}
