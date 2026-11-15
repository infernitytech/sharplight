namespace Sharplight.Studio.Services.Updates;

/// <summary>The outcome of one update check.</summary>
/// <param name="Offer">The newer version; null when Studio is up to date or the check failed.</param>
/// <param name="Error">What went wrong (network, bad feed); null when the check worked.</param>
internal sealed record UpdateCheckResult(UpdateOffer? Offer, string? Error)
{
    /// <summary>The check worked and found nothing newer.</summary>
    public static UpdateCheckResult UpToDate { get; } = new(null, null);

    /// <summary>The check found <paramref name="offer"/>.</summary>
    public static UpdateCheckResult Found(UpdateOffer offer) => new(offer, null);

    /// <summary>The check failed with <paramref name="error"/>.</summary>
    public static UpdateCheckResult Failed(string error) => new(null, error);
}
