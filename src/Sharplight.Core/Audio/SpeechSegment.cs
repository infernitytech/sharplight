namespace Sharplight.Audio;

/// <summary>A stretch of speech found by a voice activity detector.</summary>
/// <param name="StartSample">The first sample of the segment, at the sample rate of the analyzed audio.</param>
/// <param name="EndSample">The sample after the last of the segment, at the sample rate of the analyzed audio.</param>
/// <param name="SampleRate">The sample rate the sample positions refer to.</param>
public readonly record struct SpeechSegment(long StartSample, long EndSample, int SampleRate)
{
    /// <summary>The start of the segment.</summary>
    public TimeSpan Start => TimeSpan.FromSeconds((double)StartSample / SampleRate);

    /// <summary>The end of the segment.</summary>
    public TimeSpan End => TimeSpan.FromSeconds((double)EndSample / SampleRate);

    /// <summary>The length of the segment.</summary>
    public TimeSpan Duration => End - Start;
}
