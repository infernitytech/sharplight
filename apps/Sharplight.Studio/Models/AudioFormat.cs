using System.Globalization;

namespace Sharplight.Studio.Models;

/// <summary>Text forms of audio lengths.</summary>
internal static class AudioFormat
{
    /// <summary>A clip length: <c>3.2 s</c> below a minute, <c>2:05</c> up to an hour, <c>1:02:05</c> beyond.</summary>
    /// <param name="length">The length.</param>
    public static string Length(TimeSpan length)
    {
        if (length.TotalMinutes < 1)
        {
            return string.Create(CultureInfo.InvariantCulture, $"{length.TotalSeconds:0.0} s");
        }

        return length.TotalHours >= 1
            ? string.Create(CultureInfo.InvariantCulture, $"{(int)length.TotalHours}:{length.Minutes:00}:{length.Seconds:00}")
            : string.Create(CultureInfo.InvariantCulture, $"{length.Minutes}:{length.Seconds:00}");
    }
}
