namespace Sharplight.Runtime;

public sealed record ProviderAvailability(bool IsAvailable, string Reason)
{
    public static ProviderAvailability Available(string detail = "") => new(true, detail);
    public static ProviderAvailability Unavailable(string reason) => new(false, reason);
}
