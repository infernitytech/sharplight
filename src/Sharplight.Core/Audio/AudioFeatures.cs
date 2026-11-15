namespace Sharplight.Audio;

/// <summary>Frame-level features of a recording: channel-major [<see cref="Channels"/>, <see cref="Frames"/>] f32.</summary>
internal sealed class AudioFeatures
{
    /// <summary>Wraps feature values.</summary>
    /// <param name="values">Channel-major values; not copied.</param>
    /// <param name="channels">Feature channels.</param>
    /// <param name="frames">Frames.</param>
    public AudioFeatures(float[] values, int channels, int frames)
    {
        if (values.Length != (long)channels * frames)
        {
            throw new ArgumentException($"{values.Length} values are not [{channels}, {frames}].");
        }

        Values = values;
        Channels = channels;
        Frames = frames;
    }

    /// <summary>Channel-major values.</summary>
    public float[] Values { get; }

    /// <summary>Feature channels.</summary>
    public int Channels { get; }

    /// <summary>Frames.</summary>
    public int Frames { get; }

    /// <summary>
    /// Copies frames <c>[first, first + count)</c> into a channel-major [<see cref="Channels"/>, <paramref name="count"/>] window; frames past
    /// the end are filled with <paramref name="fill"/>.
    /// </summary>
    /// <param name="first">The first frame.</param>
    /// <param name="count">Frames per window.</param>
    /// <param name="destination">Receives Channels · count values.</param>
    /// <param name="fill">The value of frames past the end.</param>
    public void CopyWindow(int first, int count, Span<float> destination, float fill = 0f)
    {
        int available = Math.Clamp(Frames - first, 0, count);
        for (int channel = 0; channel < Channels; channel++)
        {
            var row = destination.Slice(channel * count, count);
            Values.AsSpan((channel * Frames) + first, available).CopyTo(row);
            row[available..].Fill(fill);
        }
    }
}
