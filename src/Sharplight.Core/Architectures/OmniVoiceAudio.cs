namespace Sharplight.Architectures;

/// <summary>
/// OmniVoice's audio post-processing as the official package does it: silence removal (the package's <c>pydub</c>-based <c>remove_silence</c>: a 16-bit quantization, windows of integer RMS
/// against -50 dBFS, middle silences longer than a limit cut to the limit, the edges trimmed), fades and padding, and cross-fading chunks.
/// </summary>
internal static class OmniVoiceAudio
{
    private const double SilenceThresholdDb = -50;
    private const double MaxAmplitude = 32768;

    /// <summary>Removes middle silences longer than <paramref name="middleSilence"/> milliseconds and trims the edges, keeping the given margins.</summary>
    /// <param name="audio">The mono signal.</param>
    /// <param name="sampleRate">Its sample rate.</param>
    /// <param name="middleSilence">Middle-silence limit in ms (0 skips the middle).</param>
    /// <param name="leadSilence">Leading silence kept, ms.</param>
    /// <param name="trailSilence">Trailing silence kept, ms.</param>
    public static float[] RemoveSilence(float[] audio, int sampleRate, int middleSilence, int leadSilence, int trailSilence)
    {
        ArgumentNullException.ThrowIfNull(audio);
        var wave = new Segment(Quantize(audio), sampleRate);
        if (middleSilence > 0)
        {
            var parts = SplitOnSilence(wave, middleSilence, SilenceThresholdDb, middleSilence, 10);
            var joined = new List<short>();
            foreach (var part in parts)
            {
                joined.AddRange(part.Samples);
            }

            wave = new Segment([.. joined], sampleRate);
        }

        var shortened = RemoveEdges(wave, leadSilence, trailSilence);
        var result = new float[shortened.Samples.Length];
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = shortened.Samples[i] / 32768f;
        }

        return result;
    }

    /// <summary>Applies a linear fade in and out and pads silence on both sides.</summary>
    /// <param name="audio">The mono signal.</param>
    /// <param name="padDuration">Silence on each side, seconds.</param>
    /// <param name="fadeDuration">Length of each fade, seconds.</param>
    /// <param name="sampleRate">The sample rate.</param>
    public static float[] FadeAndPad(float[] audio, double padDuration, double fadeDuration, int sampleRate)
    {
        ArgumentNullException.ThrowIfNull(audio);
        if (audio.Length == 0)
        {
            return audio;
        }

        int fade = (int)(fadeDuration * sampleRate);
        int pad = (int)(padDuration * sampleRate);
        var processed = (float[])audio.Clone();
        int k = Math.Min(fade, processed.Length / 2);
        if (fade > 0 && k > 0)
        {
            for (int i = 0; i < k; i++)
            {
                float ramp = k == 1 ? 0f : i / (float)(k - 1);
                processed[i] *= ramp;
                processed[processed.Length - 1 - i] *= k == 1 ? 1f : ramp;
            }
        }

        if (pad <= 0)
        {
            return processed;
        }

        var padded = new float[processed.Length + (2 * pad)];
        processed.CopyTo(padded, pad);
        return padded;
    }

    /// <summary>Joins chunks with a silence gap and a fade out and in at each boundary.</summary>
    /// <param name="chunks">The chunks.</param>
    /// <param name="sampleRate">The sample rate.</param>
    /// <param name="silenceDuration">The total gap, seconds (a third of it is silence, a third each fade).</param>
    public static float[] CrossFade(IReadOnlyList<float[]> chunks, int sampleRate, double silenceDuration = 0.3)
    {
        ArgumentNullException.ThrowIfNull(chunks);
        if (chunks.Count == 1)
        {
            return chunks[0];
        }

        int total = (int)(silenceDuration * sampleRate);
        int fade = total / 3;
        var merged = new List<float>(chunks[0]);
        for (int c = 1; c < chunks.Count; c++)
        {
            int fadeOut = Math.Min(fade, merged.Count);
            for (int i = 0; i < fadeOut; i++)
            {
                float ramp = fadeOut == 1 ? 1f : 1f - (i / (float)(fadeOut - 1));
                merged[merged.Count - fadeOut + i] *= ramp;
            }

            merged.AddRange(new float[fade]);
            var chunk = (float[])chunks[c].Clone();
            int fadeIn = Math.Min(fade, chunk.Length);
            for (int i = 0; i < fadeIn; i++)
            {
                chunk[i] *= fadeIn == 1 ? 0f : i / (float)(fadeIn - 1);
            }

            merged.AddRange(chunk);
        }

        return [.. merged];
    }

    private static short[] Quantize(float[] audio)
    {
        var samples = new short[audio.Length];
        for (int i = 0; i < samples.Length; i++)
        {
            samples[i] = (short)Math.Clamp(audio[i] * 32768.0f, -32768f, 32767f);
        }

        return samples;
    }

    private static Segment RemoveEdges(Segment wave, int leadSilence, int trailSilence)
    {
        int start = Math.Max(0, LeadingSilence(wave) - leadSilence);
        var trimmed = wave.Slice(start, wave.LengthMs);
        var reversed = trimmed.Reverse();
        start = Math.Max(0, LeadingSilence(reversed) - trailSilence);
        return reversed.Slice(start, reversed.LengthMs).Reverse();
    }

    // Milliseconds of leading silence (windows of 10 ms below the threshold).
    private static int LeadingSilence(Segment sound)
    {
        int trim = 0;
        while (trim < sound.LengthMs && sound.Slice(trim, trim + 10).DbFs < SilenceThresholdDb)
        {
            trim += 10;
        }

        return Math.Min(trim, sound.LengthMs);
    }

    private static List<(int Start, int End)> DetectSilence(Segment wave, int minimumLength, double threshold, int step)
    {
        int length = wave.LengthMs;
        var ranges = new List<(int, int)>();
        if (length < minimumLength)
        {
            return ranges;
        }

        double limit = Math.Pow(10, threshold / 20) * MaxAmplitude;
        var starts = new List<int>();
        int last = length - minimumLength;
        for (int i = 0; i <= last; i += step)
        {
            starts.Add(i);
        }

        if (last % step != 0)
        {
            starts.Add(last);
        }

        var silent = new List<int>();
        foreach (int i in starts)
        {
            if (wave.Slice(i, i + minimumLength).Rms <= limit)
            {
                silent.Add(i);
            }
        }

        if (silent.Count == 0)
        {
            return ranges;
        }

        int previous = silent[0];
        int rangeStart = previous;
        for (int n = 1; n < silent.Count; n++)
        {
            int current = silent[n];
            bool continuous = current == previous + step;
            bool gap = current > previous + minimumLength;
            if (!continuous && gap)
            {
                ranges.Add((rangeStart, previous + minimumLength));
                rangeStart = current;
            }

            previous = current;
        }

        ranges.Add((rangeStart, previous + minimumLength));
        return ranges;
    }

    private static List<(int Start, int End)> DetectNonSilent(Segment wave, int minimumLength, double threshold, int step)
    {
        var silent = DetectSilence(wave, minimumLength, threshold, step);
        int length = wave.LengthMs;
        if (silent.Count == 0)
        {
            return [(0, length)];
        }

        if (silent[0].Start == 0 && silent[0].End == length)
        {
            return [];
        }

        var ranges = new List<(int Start, int End)>();
        int previousEnd = 0;
        foreach (var (start, end) in silent)
        {
            ranges.Add((previousEnd, start));
            previousEnd = end;
        }

        if (silent[^1].End != length)
        {
            ranges.Add((previousEnd, length));
        }

        if (ranges[0] == (0, 0))
        {
            ranges.RemoveAt(0);
        }

        return ranges;
    }

    private static List<Segment> SplitOnSilence(Segment wave, int minimumLength, double threshold, int keep, int step)
    {
        var ranges = DetectNonSilent(wave, minimumLength, threshold, step).Select(r => new[] { r.Start - keep, r.End + keep }).ToList();
        for (int i = 0; i + 1 < ranges.Count; i++)
        {
            int lastEnd = ranges[i][1];
            int nextStart = ranges[i + 1][0];
            if (nextStart < lastEnd)
            {
                ranges[i][1] = (int)Math.Floor((lastEnd + nextStart) / 2.0);
                ranges[i + 1][0] = ranges[i][1];
            }
        }

        return [.. ranges.Select(r => wave.Slice(Math.Max(r[0], 0), Math.Min(r[1], wave.LengthMs)))];
    }

    /// <summary>A mono 16-bit signal addressed in milliseconds (pydub's <c>AudioSegment</c> semantics).</summary>
    private sealed class Segment
    {
        private readonly int _rate;

        public Segment(short[] samples, int rate)
        {
            Samples = samples;
            _rate = rate;
        }

        public short[] Samples { get; }

        public int LengthMs => (int)Math.Round(1000.0 * Samples.Length / _rate, MidpointRounding.ToEven);

        public int Rms
        {
            get
            {
                if (Samples.Length == 0)
                {
                    return 0;
                }

                double sum = 0;
                foreach (short sample in Samples)
                {
                    sum += (double)sample * sample;
                }

                return (int)Math.Sqrt(sum / Samples.Length);
            }
        }

        public double DbFs
        {
            get
            {
                int rms = Rms;
                return rms == 0 ? double.NegativeInfinity : 20 * Math.Log10(rms / MaxAmplitude);
            }
        }

        public Segment Slice(int startMs, int endMs)
        {
            int length = LengthMs;
            int start = Position(Math.Min(startMs, length));
            int end = Position(Math.Min(endMs, length));
            start = Math.Min(start, Samples.Length);
            end = Math.Min(end, Samples.Length);
            return new Segment(end > start ? Samples[start..end] : [], _rate);
        }

        public Segment Reverse() => new([.. Samples.Reverse()], _rate);

        private int Position(int milliseconds) => (int)(milliseconds * (_rate / 1000.0));
    }
}
