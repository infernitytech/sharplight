using System.Globalization;
using System.Text;

namespace Sharplight.Studio.Models;

/// <summary>
/// A transcript as the Transcribe page stores and exports it (<c>transcript.json</c> of a run): the text, the detected language and
/// the timestamped segments, with the plain-text, SubRip (<c>.srt</c>) and WebVTT (<c>.vtt</c>) renderings.
/// </summary>
internal sealed class TranscriptFile
{
    /// <summary>The whole text.</summary>
    public string Text { get; set; } = "";

    /// <summary>The language (code) the text is in, when known.</summary>
    public string? Language { get; set; }

    /// <summary>The length of the audio, in seconds.</summary>
    public double DurationSeconds { get; set; }

    /// <summary>The segments in order.</summary>
    public List<TranscriptSegmentRecord> Segments { get; set; } = [];

    /// <summary>The distinct speaker labels in order of first appearance (empty when no segment has a speaker).</summary>
    public IReadOnlyList<string> Speakers => [.. Segments.Select(segment => segment.SpeakerLabel).Where(label => label.Length > 0).Distinct()];

    /// <summary>
    /// The plain text: the segments' text joined by single spaces (the engine's text when there are no segments). With speakers, one line per turn
    /// (consecutive segments of the same speaker are joined) that starts with the speaker's label.
    /// </summary>
    public string ToText()
    {
        if (Segments.Count == 0)
        {
            return Text.Trim();
        }

        if (Speakers.Count == 0)
        {
            return string.Join(' ', Segments.Select(segment => segment.Text.Trim()).Where(text => text.Length > 0));
        }

        var lines = new List<string>();
        string current = "";
        var turn = new StringBuilder();
        foreach (var segment in Segments)
        {
            string text = segment.Text.Trim();
            if (text.Length == 0)
            {
                continue;
            }

            if (turn.Length > 0 && segment.SpeakerLabel != current)
            {
                lines.Add(Line(current, turn.ToString()));
                turn.Clear();
            }

            current = segment.SpeakerLabel;
            turn.Append(turn.Length > 0 ? " " : "").Append(text);
        }

        if (turn.Length > 0)
        {
            lines.Add(Line(current, turn.ToString()));
        }

        return string.Join('\n', lines);
    }

    private static string Line(string speaker, string text) => speaker.Length == 0 ? text : $"{speaker}: {text}";

    /// <summary>The transcript as SubRip subtitles.</summary>
    public string ToSrt()
    {
        var builder = new StringBuilder();
        for (int i = 0; i < Segments.Count; i++)
        {
            var segment = Segments[i];
            builder.Append((i + 1).ToString(CultureInfo.InvariantCulture)).Append('\n')
                .Append(Clock(segment.Start, ',')).Append(" --> ").Append(Clock(segment.End, ',')).Append('\n')
                .Append(Line(segment.SpeakerLabel, segment.Text.Trim())).Append("\n\n");
        }

        return builder.ToString();
    }

    /// <summary>The transcript as WebVTT subtitles.</summary>
    public string ToVtt()
    {
        var builder = new StringBuilder("WEBVTT\n\n");
        foreach (var segment in Segments)
        {
            builder.Append(Clock(segment.Start, '.')).Append(" --> ").Append(Clock(segment.End, '.')).Append('\n')
                .Append(segment.SpeakerLabel.Length == 0 ? "" : $"<v {segment.SpeakerLabel}>").Append(segment.Text.Trim()).Append("\n\n");
        }

        return builder.ToString();
    }

    /// <summary>A time as <c>hh:mm:ss,mmm</c> (or with another fraction separator).</summary>
    /// <param name="seconds">Seconds.</param>
    /// <param name="separator">The separator before the milliseconds.</param>
    public static string Clock(double seconds, char separator)
    {
        var time = TimeSpan.FromSeconds(Math.Max(0, seconds));
        return string.Create(CultureInfo.InvariantCulture, $"{(int)time.TotalHours:00}:{time.Minutes:00}:{time.Seconds:00}{separator}{time.Milliseconds:000}");
    }
}
