using Sharplight.Audio;

namespace Sharplight.AudioModels.Tests;

/// <summary>The probability-to-segment step of voice activity detection, on synthetic probabilities.</summary>
public class SpeechSegmentationTests
{
    private static float[] Probabilities(params (int Frames, float Value)[] runs) => [.. runs.SelectMany(r => Enumerable.Repeat(r.Value, r.Frames))];

    [Fact]
    public void One_run_of_speech_becomes_one_padded_segment()
    {
        var probabilities = Probabilities((20, 0.01f), (40, 0.9f), (40, 0.01f));
        var segments = SpeechSegmentation.FromProbabilities(probabilities, 512, 16000, probabilities.Length * 512L);
        var segment = Assert.Single(segments);
        Assert.Equal((20 * 512) - 480, segment.StartSample);
        Assert.Equal(((20 + 40) * 512) + 480, segment.EndSample);
    }

    [Fact]
    public void A_short_blip_is_dropped_and_a_short_silence_does_not_split()
    {
        var blip = Probabilities((20, 0.01f), (3, 0.9f), (20, 0.01f));
        Assert.Empty(SpeechSegmentation.FromProbabilities(blip, 512, 16000, blip.Length * 512L));
        var gap = Probabilities((5, 0.01f), (30, 0.9f), (2, 0.01f), (30, 0.9f), (10, 0.01f));
        Assert.Single(SpeechSegmentation.FromProbabilities(gap, 512, 16000, gap.Length * 512L));
    }

    [Fact]
    public void An_overlong_segment_is_split_at_its_longest_silence()
    {
        var probabilities = Probabilities((10, 0.9f), (4, 0.01f), (30, 0.9f), (8, 0.01f), (30, 0.9f), (10, 0.01f));
        var options = new VoiceActivityOptions { MaxSpeechDuration = TimeSpan.FromSeconds(1.5), MinSilenceDuration = TimeSpan.FromMilliseconds(500) };
        var segments = SpeechSegmentation.FromProbabilities(probabilities, 512, 16000, probabilities.Length * 512L, options);
        Assert.True(segments.Count >= 2);
        Assert.All(segments, s => Assert.True(s.Duration < TimeSpan.FromSeconds(2.2)));
    }
}
