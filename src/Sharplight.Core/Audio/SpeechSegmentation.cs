namespace Sharplight.Audio;

/// <summary>Turns per-frame speech probabilities into speech segments with hysteresis, minimum durations, padding and splitting of overlong segments (the algorithm of Silero VAD's reference implementation).</summary>
public static class SpeechSegmentation
{
    /// <summary>Segments the probabilities.</summary>
    /// <param name="probabilities">One speech probability per frame.</param>
    /// <param name="frameSamples">The samples one frame covers.</param>
    /// <param name="sampleRate">The sample rate the probabilities were computed at.</param>
    /// <param name="totalSamples">The number of samples of the analyzed audio at <paramref name="sampleRate"/>.</param>
    /// <param name="options">Segmentation settings, or <see langword="null"/> for the defaults.</param>
    /// <returns>The segments in samples at <paramref name="sampleRate"/>.</returns>
    public static IReadOnlyList<SpeechSegment> FromProbabilities(ReadOnlySpan<float> probabilities, int frameSamples, int sampleRate, long totalSamples, VoiceActivityOptions? options = null)
    {
        options ??= new VoiceActivityOptions();
        double minSpeech = sampleRate * options.MinSpeechDuration.TotalSeconds;
        double pad = sampleRate * options.SpeechPad.TotalSeconds;
        double maxSpeech = options.MaxSpeechDuration == TimeSpan.MaxValue ? double.PositiveInfinity : (sampleRate * options.MaxSpeechDuration.TotalSeconds) - frameSamples - (2 * pad);
        double minSilence = sampleRate * options.MinSilenceDuration.TotalSeconds;
        double minSilenceAtMax = sampleRate * options.MinSilenceAtMaxSpeech.TotalSeconds;
        float threshold = options.Threshold;
        float negative = options.NegativeThreshold ?? MathF.Max(threshold - 0.15f, 0.01f);

        bool triggered = false;
        var speeches = new List<(long Start, long End)>();
        long start = 0;
        bool hasCurrent = false;
        long tempEnd = 0;
        long previousEnd = 0;
        long nextStart = 0;
        var possibleEnds = new List<(long End, long Duration)>();

        for (int i = 0; i < probabilities.Length; i++)
        {
            float probability = probabilities[i];
            long current = (long)frameSamples * i;
            if (probability >= threshold && tempEnd != 0)
            {
                long silence = current - tempEnd;
                if (silence > minSilenceAtMax)
                {
                    possibleEnds.Add((tempEnd, silence));
                }

                tempEnd = 0;
                if (nextStart < previousEnd)
                {
                    nextStart = current;
                }
            }

            if (probability >= threshold && !triggered)
            {
                triggered = true;
                start = current;
                hasCurrent = true;
                continue;
            }

            if (triggered && current - start > maxSpeech)
            {
                if (options.UseLongestSilenceAtMaxSpeech && possibleEnds.Count > 0)
                {
                    var longest = possibleEnds[0];
                    foreach (var candidate in possibleEnds)
                    {
                        if (candidate.Duration > longest.Duration)
                        {
                            longest = candidate;
                        }
                    }

                    previousEnd = longest.End;
                    speeches.Add((start, previousEnd));
                    hasCurrent = false;
                    nextStart = previousEnd + longest.Duration;
                    if (nextStart < previousEnd + current)
                    {
                        start = nextStart;
                        hasCurrent = true;
                    }
                    else
                    {
                        triggered = false;
                    }

                    previousEnd = nextStart = tempEnd = 0;
                    possibleEnds.Clear();
                }
                else if (previousEnd != 0)
                {
                    speeches.Add((start, previousEnd));
                    hasCurrent = false;
                    if (nextStart < previousEnd)
                    {
                        triggered = false;
                    }
                    else
                    {
                        start = nextStart;
                        hasCurrent = true;
                    }

                    previousEnd = nextStart = tempEnd = 0;
                    possibleEnds.Clear();
                }
                else
                {
                    speeches.Add((start, current));
                    hasCurrent = false;
                    previousEnd = nextStart = tempEnd = 0;
                    triggered = false;
                    possibleEnds.Clear();
                    continue;
                }
            }

            if (probability < negative && triggered)
            {
                if (tempEnd == 0)
                {
                    tempEnd = current;
                }

                long silenceNow = current - tempEnd;
                if (!options.UseLongestSilenceAtMaxSpeech && silenceNow > minSilenceAtMax)
                {
                    previousEnd = tempEnd;
                }

                if (silenceNow < minSilence)
                {
                    continue;
                }

                if (tempEnd - start > minSpeech)
                {
                    speeches.Add((start, tempEnd));
                }

                hasCurrent = false;
                previousEnd = nextStart = tempEnd = 0;
                triggered = false;
                possibleEnds.Clear();
            }
        }

        if (hasCurrent && totalSamples - start > minSpeech)
        {
            speeches.Add((start, totalSamples));
        }

        var result = new (long Start, long End)[speeches.Count];
        speeches.CopyTo(result);
        for (int i = 0; i < result.Length; i++)
        {
            if (i == 0)
            {
                result[i].Start = (long)Math.Max(0, result[i].Start - pad);
            }

            if (i != result.Length - 1)
            {
                long silence = result[i + 1].Start - result[i].End;
                if (silence < 2 * pad)
                {
                    long half = (long)Math.Floor(silence / 2.0);
                    result[i].End += half;
                    result[i + 1].Start = Math.Max(0, result[i + 1].Start - half);
                }
                else
                {
                    result[i].End = (long)Math.Min(totalSamples, result[i].End + pad);
                    result[i + 1].Start = (long)Math.Max(0, result[i + 1].Start - pad);
                }
            }
            else
            {
                result[i].End = (long)Math.Min(totalSamples, result[i].End + pad);
            }
        }

        return [.. result.Select(r => new SpeechSegment(r.Start, r.End, sampleRate))];
    }
}
