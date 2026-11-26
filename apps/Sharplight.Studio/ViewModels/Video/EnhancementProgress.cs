using System.Globalization;
using Sharplight.VideoEnhancement;

namespace Sharplight.Studio.ViewModels.Video;

/// <summary>
/// The progress line of an Enhance run: the stage of the batch, the frames finished out of the clip's (also as playback time), the speed and
/// the time left. The speed is measured from the first finished batch on, so the compilation of the first batch does not skew the estimate.
/// </summary>
/// <param name="expectedFrames">The frames the input has (the probe's count; 1 for an image).</param>
/// <param name="framesPerSecond">The input's playback rate, 0 when unknown (no playback times then).</param>
internal sealed class EnhancementProgress(long expectedFrames, double framesPerSecond)
{
    private (long Frames, TimeSpan Elapsed)? _firstBatch;

    /// <summary>The line and the completed fraction for an update.</summary>
    /// <param name="update">The update.</param>
    /// <param name="elapsed">The time since the run started.</param>
    /// <returns>The text, and the fraction done (0-1).</returns>
    public (string Text, double Fraction) Describe(VideoEnhancementUpdate update, TimeSpan elapsed)
    {
        long total = Math.Max(1, update.TotalFrames ?? expectedFrames);
        long finished = update.FinishedFrames;
        string phase = update.Phase switch
        {
            VideoEnhancementPhase.Encoding => "encoding",
            VideoEnhancementPhase.Restoring => "restoring",
            VideoEnhancementPhase.Decoding => update.Frames is null ? "decoding" : "decoded",
            _ => "Done",
        };
        if (update.Frames is { Count: > 0 })
        {
            _firstBatch ??= (finished, elapsed);
        }

        if (total <= 1)
        {
            return (phase, update.Phase == VideoEnhancementPhase.Completed ? 1 : StageFraction(update.Phase, update.Frames is not null));
        }

        // Within a batch the stages move the bar a little, so it does not sit still for the length of one batch.
        long batchFrames = Math.Max(1, update.TotalBatches > 0 ? total / update.TotalBatches : total);
        double fraction = update.Phase == VideoEnhancementPhase.Completed
            ? 1
            : Math.Min(1, (finished + (update.Frames is null ? StageFraction(update.Phase, false) * batchFrames : 0)) / (double)total);
        var parts = new List<string>
        {
            update.TotalBatches > 1 ? $"Batch {update.Batch + 1} of {update.TotalBatches}: {phase}" : phase,
            string.Create(CultureInfo.InvariantCulture, $"{finished:N0} of {total:N0} frames"),
        };
        if (framesPerSecond > 0)
        {
            parts.Add($"{Clock(TimeSpan.FromSeconds(finished / framesPerSecond))} of {Clock(TimeSpan.FromSeconds(total / framesPerSecond))}");
        }

        if (_firstBatch is { } first && finished > first.Frames && elapsed > first.Elapsed && update.Phase != VideoEnhancementPhase.Completed)
        {
            double rate = (finished - first.Frames) / (elapsed - first.Elapsed).TotalSeconds;
            parts.Add(string.Create(CultureInfo.InvariantCulture, $"{rate:0.0#} frames/s"));
            parts.Add($"about {Clock(TimeSpan.FromSeconds(Math.Max(0, total - finished) / rate))} left");
        }

        return (string.Join(" · ", parts), fraction);
    }

    /// <summary>A duration as h:mm:ss, or m:ss under an hour.</summary>
    /// <param name="time">The duration.</param>
    internal static string Clock(TimeSpan time) => time.TotalHours >= 1
        ? string.Create(CultureInfo.InvariantCulture, $"{(int)time.TotalHours}:{time.Minutes:00}:{time.Seconds:00}")
        : string.Create(CultureInfo.InvariantCulture, $"{time.Minutes}:{time.Seconds:00}");

    private static double StageFraction(VideoEnhancementPhase phase, bool decoded) => phase switch
    {
        VideoEnhancementPhase.Encoding => 0.05,
        VideoEnhancementPhase.Restoring => 0.15,
        VideoEnhancementPhase.Decoding => decoded ? 1.0 : 0.85,
        _ => 1.0,
    };
}
