using Avalonia;
using Sharplight.Audio;
using Sharplight.Studio.Models;

namespace Sharplight.Studio.ViewModels.Audio;

/// <summary>One speech segment of the voice activity timeline: its times and where its bar lies in a timeline of <see cref="TimelineWidth"/> pixels.</summary>
internal sealed class SpeechSegmentViewModel
{
    /// <summary>The width of the timeline in pixels.</summary>
    public const double TimelineWidth = 560;

    /// <param name="segment">The segment.</param>
    /// <param name="total">The length of the whole recording.</param>
    public SpeechSegmentViewModel(SpeechSegment segment, TimeSpan total)
    {
        double length = Math.Max(0.001, total.TotalSeconds);
        Start = segment.Start;
        End = segment.End;
        Offset = new Thickness(segment.Start.TotalSeconds / length * TimelineWidth, 0, 0, 0);
        BarWidth = Math.Max(2, segment.Duration.TotalSeconds / length * TimelineWidth);
    }

    /// <summary>The start of the segment.</summary>
    public TimeSpan Start { get; }

    /// <summary>The end of the segment.</summary>
    public TimeSpan End { get; }

    /// <summary>The label of the segment: start, end and length.</summary>
    public string Text => $"{AudioFormat.Length(Start)} – {AudioFormat.Length(End)}  ({(End - Start).TotalSeconds:0.00} s)";

    /// <summary>The left margin of the segment's bar in the timeline.</summary>
    public Thickness Offset { get; }

    /// <summary>The width of the segment's bar in the timeline.</summary>
    public double BarWidth { get; }
}
