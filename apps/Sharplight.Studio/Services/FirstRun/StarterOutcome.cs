namespace Sharplight.Studio.Services.FirstRun;

/// <summary>What the recommendation found.</summary>
internal enum StarterOutcome
{
    /// <summary>At least a chat model is recommended.</summary>
    Recommended,

    /// <summary>The feed lists nothing usable (offline without a saved list, empty channel, nothing installable).</summary>
    NoPackages,

    /// <summary>The feed has chat models, but none fits the memory or the disk of this machine.</summary>
    NothingFits,
}
