namespace Sharplight.Studio.Models;

/// <summary>How models stay resident: memory reserve, limits, idle unloading and on-demand loading.</summary>
internal sealed record ResidencySettings
{
    /// <summary>Device memory kept free for other applications, in MiB (a model loads only when it fits with this to spare).</summary>
    public int ReserveMiB { get; init; } = 1024;

    /// <summary>At most this many models resident at once (0 = no limit); loading another evicts the least recently used idle one.</summary>
    public int MaxLoadedModels { get; init; }

    /// <summary>Unload models that served nothing for this many minutes (0 = never); they reload on their next request.</summary>
    public int IdleUnloadMinutes { get; init; }

    /// <summary>Models loaded by hand are pinned: never evicted or unloaded for idleness.</summary>
    public bool PinManualLoads { get; init; } = true;

    /// <summary>A chat or API request naming a library model that is not resident loads it with the default load settings (ollama-style).</summary>
    public bool LoadOnDemand { get; init; }
}
