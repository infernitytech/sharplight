namespace Sharplight.Runtime;

public readonly record struct OpSupport(bool IsSupported, string? Reason = null)
{
    public static OpSupport Yes => new(true);
    public static OpSupport No(string reason) => new(false, reason);
}
