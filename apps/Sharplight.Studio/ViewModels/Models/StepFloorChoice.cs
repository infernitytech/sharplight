using Sharplight.Llm.Serving;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>A choice of when the floor of the engine's adaptive step budget applies (<see cref="StepFloorPolicy"/>).</summary>
/// <param name="Value">The policy, or null for the engine's default.</param>
/// <param name="Name">Shown in the picker.</param>
/// <param name="Description">What the choice trades, for the tooltip.</param>
internal sealed record StepFloorChoice(StepFloorPolicy? Value, string Name, string Description)
{
    /// <summary>The choices, the engine's default first.</summary>
    public static IReadOnlyList<StepFloorChoice> All { get; } =
    [
        new(null, "Default (waves)", "The engine's default: the floor applies only while no sequence decodes."),
        new(StepFloorPolicy.Waves, "Waves", "The floor applies only while no sequence decodes: prompts admitted first stream sooner."),
        new(StepFloorPolicy.Always, "Always", "The floor applies to every step: prompt waves prefill in larger steps."),
        new(StepFloorPolicy.Full, "Full", "While a sequence decodes every step may take the whole budget: best throughput, longest time to first token."),
    ];
}
