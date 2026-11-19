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

    /// <summary>The speaker's label (<c>Speaker 1</c>), or empty when the model does not tell speakers apart.</summary>
    public string Speaker => Record.SpeakerLabel;

    /// <summary>The segment has a speaker.</summary>
    public bool HasSpeaker => Record.SpeakerLabel.Length > 0;

    /// <summary>The index of the speaker's colour in <see cref="SpeakerPalette"/> (the order of first appearance is set by the page).</summary>
    public int ColorIndex { get; internal set; }

    /// <summary>The speaker's colour, as a hex string.</summary>
    public string Color => SpeakerPalette.Colors[ColorIndex % SpeakerPalette.Colors.Length];

    /// <summary>The speaker's colour as a brush.</summary>
    public Avalonia.Media.IBrush Brush => Avalonia.Media.SolidColorBrush.Parse(Color);

    /// <summary>The spoken text.</summary>
    public string Text => Record.Text.Trim();

    private static string Short(double seconds)
    {
        var time = TimeSpan.FromSeconds(seconds);
        return time.TotalHours >= 1 ? time.ToString(@"h\:mm\:ss\.f", System.Globalization.CultureInfo.InvariantCulture) : time.ToString(@"mm\:ss\.f", System.Globalization.CultureInfo.InvariantCulture);
    }
}
