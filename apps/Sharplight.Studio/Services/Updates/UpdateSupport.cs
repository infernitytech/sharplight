namespace Sharplight.Studio.Services.Updates;

/// <summary>Whether this run of Studio can update itself, and why not.</summary>
/// <param name="IsSupported">Updates can be checked for and applied.</param>
/// <param name="Reason">Why not, shown to the user; null when supported.</param>
internal sealed record UpdateSupport(bool IsSupported, string? Reason)
{
    /// <summary>Updates work.</summary>
    public static UpdateSupport Supported { get; } = new(true, null);

    /// <summary>Updates do not work, for <paramref name="reason"/>.</summary>
    public static UpdateSupport Unsupported(string reason) => new(false, reason);
}
