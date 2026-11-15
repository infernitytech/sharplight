namespace Sharplight.Studio.Models;

/// <summary>Which repositories Discover lists with respect to gating (accepting terms and a token to download).</summary>
internal enum GatedFilter
{
    /// <summary>Gated and open repositories.</summary>
    Any,

    /// <summary>Only open repositories.</summary>
    Hide,

    /// <summary>Only gated repositories.</summary>
    Only,
}
