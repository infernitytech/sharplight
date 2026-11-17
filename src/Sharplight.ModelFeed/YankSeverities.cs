namespace Sharplight.ModelFeed;

/// <summary>Severities of <see cref="PackageYank"/>.</summary>
public static class YankSeverities
{
    /// <summary>Wrong or degraded output.</summary>
    public const string Quality = "quality";

    /// <summary>Licensing or legal issue.</summary>
    public const string Legal = "legal";

    /// <summary>Security issue: hosts alert and may block loads.</summary>
    public const string Security = "security";
}
