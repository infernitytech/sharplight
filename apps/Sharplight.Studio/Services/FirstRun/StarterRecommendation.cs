namespace Sharplight.Studio.Services.FirstRun;

/// <summary>The starter models recommended for a machine, or why there are none.</summary>
internal sealed record StarterRecommendation
{
    /// <summary>What was found.</summary>
    public StarterOutcome Outcome { get; init; }

    /// <summary>The recommended packages, the chat model first; empty unless <see cref="Outcome"/> is <see cref="StarterOutcome.Recommended"/>.</summary>
    public IReadOnlyList<StarterItem> Items { get; init; } = [];

    /// <summary>The recommendation is sized to the CPU because the machine has no usable GPU.</summary>
    public bool CpuOnly { get; init; }

    /// <summary>The memory a chat model may need on this machine.</summary>
    public long MemoryBudgetBytes { get; init; }

    /// <summary>Explains an empty result, or notes what was left out; empty otherwise.</summary>
    public string Message { get; init; } = "";

    /// <summary>The first message suggested for the new chat.</summary>
    public string ExamplePrompt { get; init; } = "";
}
