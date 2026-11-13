namespace Sharplight.Runtime;

/// <summary>
/// Whether a device can execute a graph node.
/// </summary>
/// <param name="IsSupported">True when it can.</param>
/// <param name="Reason">Why not, for diagnostics.</param>
public readonly record struct OpSupport(bool IsSupported, string? Reason = null)
{
    /// <summary>The node is supported.</summary>
    public static OpSupport Yes => new(true);
    /// <summary>The node is not supported.</summary>
    public static OpSupport No(string reason) => new(false, reason);
}
