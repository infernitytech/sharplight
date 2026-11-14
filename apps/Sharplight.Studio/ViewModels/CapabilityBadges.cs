using Sharplight.Studio.Models;

namespace Sharplight.Studio.ViewModels;

/// <summary>Which capability badges to show for a model.</summary>
internal sealed record CapabilityBadges(ModelCapabilities Capabilities)
{
    /// <summary>Vision badge.</summary>
    public bool Vision => Capabilities.HasFlag(ModelCapabilities.Vision);

    /// <summary>Reasoning badge.</summary>
    public bool Reasoning => Capabilities.HasFlag(ModelCapabilities.Reasoning);

    /// <summary>Tools badge.</summary>
    public bool Tools => Capabilities.HasFlag(ModelCapabilities.Tools);

    /// <summary>Embedding badge.</summary>
    public bool Embedding => Capabilities.HasFlag(ModelCapabilities.Embedding);

    /// <summary>"Text only" badge: a chat model without a vision tower the engine runs.</summary>
    public bool TextOnly => Capabilities.HasFlag(ModelCapabilities.Chat) && !Vision;

    /// <summary>Nothing is known (e.g. a GGUF file before it was loaded or converted).</summary>
    public bool Unknown => Capabilities == ModelCapabilities.None;

    /// <summary>Multi-token-prediction (draft head) badge.</summary>
    public bool DraftHead => Capabilities.HasFlag(ModelCapabilities.DraftHead);
}
