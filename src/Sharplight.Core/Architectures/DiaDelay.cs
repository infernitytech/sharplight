namespace Sharplight.Architectures;

/// <summary>Dia's delay pattern: codebook <c>c</c> of a frame sits <c>delay[c]</c> rows after the frame's first codebook in the decoder's sequence.</summary>
internal static class DiaDelay
{
    /// <summary>
    /// The decoder's rows for an audio prompt: row 0 is the start token, then the prompt's frames, each codebook shifted by its delay; the table is
    /// <c>frames + 1 + max delay</c> rows long, entries past the prompt are <paramref name="pad"/> (the model fills them in).
    /// </summary>
    /// <param name="frames">The prompt's codes, one array per frame (or <see langword="null"/> for no prompt).</param>
    /// <param name="delays">The delay of every codebook.</param>
    /// <param name="bos">The start token.</param>
    /// <param name="pad">The padding token.</param>
    public static int[][] Table(IReadOnlyList<int[]>? frames, IReadOnlyList<int> delays, int bos, int pad)
    {
        int count = frames?.Count ?? 0;
        int channels = delays.Count;
        var table = new int[count + 1 + delays.Max()][];
        for (int t = 0; t < table.Length; t++)
        {
            var row = new int[channels];
            for (int c = 0; c < channels; c++)
            {
                int index = t - delays[c];
                row[c] = index <= 0 ? bos : index <= count ? frames![index - 1][c] : pad;
            }

            table[t] = row;
        }

        return table;
    }

    /// <summary>The frames <c>[start, end)</c> of a delayed sequence, the delay reverted.</summary>
    /// <param name="sequence">The sequence's rows.</param>
    /// <param name="delays">The delay of every codebook.</param>
    /// <param name="start">The first frame.</param>
    /// <param name="end">The frame after the last.</param>
    public static List<int[]> Revert(IReadOnlyList<int[]> sequence, IReadOnlyList<int> delays, int start, int end)
    {
        var frames = new List<int[]>(Math.Max(0, end - start));
        for (int f = start; f < end; f++)
        {
            var frame = new int[delays.Count];
            for (int c = 0; c < frame.Length; c++)
            {
                int row = f + delays[c];
                frame[c] = row < sequence.Count ? sequence[row][c] : 0;
            }

            frames.Add(frame);
        }

        return frames;
    }
}
