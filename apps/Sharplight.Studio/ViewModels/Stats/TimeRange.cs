namespace Sharplight.Studio.ViewModels.Stats;

/// <summary>A history range of the dashboard.</summary>
/// <param name="Name">Shown in the selector.</param>
/// <param name="Span">How far back.</param>
/// <param name="Bucket">Width of a bar in the token chart.</param>
internal sealed record TimeRange(string Name, TimeSpan Span, TimeSpan Bucket)
{
    /// <summary>The choices.</summary>
    public static IReadOnlyList<TimeRange> All { get; } =
    [
        new("1 hour", TimeSpan.FromHours(1), TimeSpan.FromMinutes(5)),
        new("24 hours", TimeSpan.FromDays(1), TimeSpan.FromHours(1)),
        new("7 days", TimeSpan.FromDays(7), TimeSpan.FromDays(1)),
        new("30 days", TimeSpan.FromDays(30), TimeSpan.FromDays(1)),
    ];
}
