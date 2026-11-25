using Sharplight.Llm.Serving;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>A choice of what the engine does with new requests when the KV cache is full (<see cref="KvAdmissionPolicy"/>).</summary>
/// <param name="Value">The policy.</param>
/// <param name="Name">Shown in the picker.</param>
/// <param name="Description">What the choice trades, for the tooltip.</param>
internal sealed record KvAdmissionChoice(KvAdmissionPolicy Value, string Name, string Description)
{
    /// <summary>The choices, the engine's default first.</summary>
    public static IReadOnlyList<KvAdmissionChoice> All { get; } =
    [
        new(KvAdmissionPolicy.Queue, "Queue new requests (default)",
            "New requests wait until running ones leave room: keeps throughput highest."),
        new(KvAdmissionPolicy.Preempt, "Preempt the newest request",
            "New requests start sooner (lower time to first token); the newest running request gives its room back and is recomputed later, at slightly lower throughput."),
    ];
}
