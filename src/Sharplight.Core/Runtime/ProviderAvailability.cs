namespace Sharplight.Runtime;

/// <summary>
/// Whether an <see cref="IExecutionProvider"/> can run on this machine.
/// </summary>
/// <param name="IsAvailable">True when devices can be opened.</param>
/// <param name="Reason">Why not (or an informational detail when available).</param>
public sealed record ProviderAvailability(bool IsAvailable, string Reason)
{
    /// <summary>Creates an "available" result with an optional detail.</summary>
    public static ProviderAvailability Available(string detail = "") => new(true, detail);
    /// <summary>Creates an "unavailable" result.</summary>
    public static ProviderAvailability Unavailable(string reason) => new(false, reason);
}
