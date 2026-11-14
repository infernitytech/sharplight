using System.Globalization;

namespace Sharplight.Studio.ViewModels;

/// <summary>"just now", "5 min ago", "yesterday", "12 Mar".</summary>
internal static class RelativeTime
{
    /// <summary>Formats <paramref name="time"/> relative to <paramref name="now"/>.</summary>
    public static string Format(DateTimeOffset time, DateTimeOffset now)
    {
        var age = now - time;
        if (age < TimeSpan.FromMinutes(1))
        {
            return "just now";
        }

        if (age < TimeSpan.FromHours(1))
        {
            return $"{(int)age.TotalMinutes} min ago";
        }

        if (age < TimeSpan.FromDays(1))
        {
            return $"{(int)age.TotalHours} h ago";
        }

        if (age < TimeSpan.FromDays(2))
        {
            return "yesterday";
        }

        return time.ToLocalTime().ToString(age < TimeSpan.FromDays(365) ? "d MMM" : "d MMM yyyy", CultureInfo.InvariantCulture);
    }
}
