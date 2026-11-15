using Sharplight.Studio.Models;

namespace Sharplight.Studio.ViewModels.Speech;

/// <summary>A timestamped segment in the transcript view.</summary>
internal sealed class TranscriptSegmentViewModel
{
    internal TranscriptSegmentViewModel(TranscriptSegmentRecord record)
    {
        Record = record;
    }

    /// <summary>The segment.</summary>
    public TranscriptSegmentRecord Record { get; }

    /// <summary>The time span, <c>00:03 - 00:07</c>.</summary>
    public string Time => $"{Short(Record.Start)} - {Short(Record.End)}";

    /// <summary>The spoken text.</summary>
    public string Text => Record.Text.Trim();

    private static string Short(double seconds)
    {
        var time = TimeSpan.FromSeconds(seconds);
        return time.TotalHours >= 1 ? time.ToString(@"h\:mm\:ss\.f", System.Globalization.CultureInfo.InvariantCulture) : time.ToString(@"mm\:ss\.f", System.Globalization.CultureInfo.InvariantCulture);
    }
}
