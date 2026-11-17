namespace Sharplight.Studio.Services.Updates;

/// <summary>The outcome of one update check.</summary>
/// <param name="Offer">The newer version; null when Studio is up to date or the check failed.</param>
/// <param name="Error">What went wrong (network, bad feed); null when the check worked.</param>
/// <param name="NoRelease">The channel has no release at all yet (its feed does not exist or is empty); not an error.</param>
internal sealed record UpdateCheckResult(UpdateOffer? Offer, string? Error, bool NoRelease = false)
{
    /// <summary>The check worked and found nothing newer.</summary>
    public static UpdateCheckResult UpToDate { get; } = new(null, null);

    /// <summary>The channel has no release yet.</summary>
    public static UpdateCheckResult NoReleaseYet { get; } = new(null, null, true);

    /// <summary>The check found <paramref name="offer"/>.</summary>
    public static UpdateCheckResult Found(UpdateOffer offer) => new(offer, null);

    /// <summary>The check failed with <paramref name="error"/>.</summary>
    public static UpdateCheckResult Failed(string error) => new(null, error);
}
