using Sharplight.Hub;

namespace Sharplight.Studio.Models;

/// <summary>The filters and sort order of the Discover page, kept in the settings.</summary>
internal sealed record DiscoverFilterSettings
{
    /// <summary>Only models of this task; null for every task.</summary>
    public ModelTask? Task { get; init; }

    /// <summary>Only repositories that provide this format; null for any (the default is GGUF, where most local language models are).</summary>
    public ModelFormat? Format { get; init; }

    /// <summary>Only repositories offering a quantization of this family (<c>Q4</c>, <c>F16</c>, <c>FP8</c>, ...); null for any.</summary>
    public string? Quantization { get; init; }

    /// <summary>Only models in this parameter range.</summary>
    public ModelSizeFilter Size { get; init; }

    /// <summary>Only models whose smallest likely download fits this machine's memory.</summary>
    public bool FitsThisMachine { get; init; }

    /// <summary>Only repositories under this license (the Hub's license id); null for any.</summary>
    public string? License { get; init; }

    /// <summary>Whether gated repositories are listed.</summary>
    public GatedFilter Gated { get; init; }

    /// <summary>The order of the results.</summary>
    public HubSortOrder Sort { get; init; } = HubSortOrder.Downloads;
}
