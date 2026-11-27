namespace Sharplight.TimeSeries;

/// <summary>A Moirai context left-padded to whole patches (<see cref="MoiraiPatches.Pad"/>) with its patch time ids (<see cref="MoiraiPatches.TimeIds"/>).</summary>
internal sealed class PaddedContext
{
    public PaddedContext(ReadOnlySpan<float> context, int patchSize)
    {
        (Values, Observed) = MoiraiPatches.Pad(context, patchSize);
        Times = MoiraiPatches.TimeIds(Observed, patchSize);
    }

    /// <summary>The values (0 where missing or padded).</summary>
    public float[] Values { get; }

    /// <summary>The observation flags.</summary>
    public bool[] Observed { get; }

    /// <summary>The time id of every patch.</summary>
    public int[] Times { get; }

    /// <summary>The patches.</summary>
    public int Rows => Times.Length;
}
