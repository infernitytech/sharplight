namespace Sharplight.Studio.Models;

/// <summary>A range of parameter counts Discover filters by. Repositories whose parameter count the Hub does not report always pass.</summary>
internal enum ModelSizeFilter
{
    /// <summary>Any size.</summary>
    Any,

    /// <summary>Under 1 billion parameters.</summary>
    Under1B,

    /// <summary>1 to 4 billion parameters.</summary>
    From1To4B,

    /// <summary>4 to 14 billion parameters.</summary>
    From4To14B,

    /// <summary>14 to 40 billion parameters.</summary>
    From14To40B,

    /// <summary>Over 40 billion parameters.</summary>
    Over40B,
}
