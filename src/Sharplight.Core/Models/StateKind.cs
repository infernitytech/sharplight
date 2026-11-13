namespace Sharplight.Models;

/// <summary>Kind of persistent state; lets runtimes manage e.g. context shifting or snapshots per kind.</summary>
public enum StateKind
{
    /// <summary>Attention key cache, [context, ...] with a symbolic context length.</summary>
    KeyCache,

    /// <summary>Attention value cache, [context, ...] with a symbolic context length.</summary>
    ValueCache,

    /// <summary>Fixed-size recurrent state (convolution window, SSM or linear-attention state).</summary>
    Recurrent,

    /// <summary>Any other persistent state.</summary>
    Other,
}
