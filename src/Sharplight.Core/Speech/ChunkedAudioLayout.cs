using Sharplight.Audio;

namespace Sharplight.Speech;

/// <summary>
/// The host side of a chunked audio tower (Qwen3-ASR): the log-mel frames of a window are cut into chunks of <c>chunkFrames</c> frames that the
/// strided convolutions process independently. The chunks are laid out back to back in one time-major feature map, each padded to
/// <see cref="PaddedFrames"/> rows, so a single sequence convolution covers all of them: the padding rows are zero at the input, and the row
/// masks zero them again after every convolution but the last, which makes every chunk see zero padding on both sides exactly as a separate
/// convolution of the chunk would. After the stem the layout gives each row its position within its chunk, the rows that hold valid frames,
/// and the attention window of every valid row.
/// </summary>
internal sealed class ChunkedAudioLayout
{
    private readonly int[] _validPerLevel;

    /// <summary>Creates the layout of a tower.</summary>
    /// <param name="chunkFrames">Frames per chunk.</param>
    /// <param name="windowFrames">Frames per attention window (a multiple of <paramref name="chunkFrames"/>).</param>
    /// <param name="convCount">Strided (stride 2, kernel 3, padding 1) convolutions.</param>
    /// <param name="melBins">Feature channels.</param>
    public ChunkedAudioLayout(int chunkFrames, int windowFrames, int convCount, int melBins)
    {
        ChunkFrames = chunkFrames;
        WindowFrames = windowFrames;
        ConvCount = convCount;
        MelBins = melBins;
        _validPerLevel = new int[convCount + 1];
        _validPerLevel[0] = chunkFrames;
        for (int i = 1; i <= convCount; i++)
        {
            _validPerLevel[i] = Subsample(_validPerLevel[i - 1]);
        }

        // Every level keeps at least one zero row after a chunk's valid rows (the padding the next convolution reads), the last level
        // only needs room for the valid rows.
        int unit = 1 << convCount;
        int padded = unit;
        while (!Fits(padded))
        {
            padded += unit;
        }

        PaddedFrames = padded;
    }

    /// <summary>Frames per chunk.</summary>
    public int ChunkFrames { get; }

    /// <summary>Frames per attention window.</summary>
    public int WindowFrames { get; }

    /// <summary>Strided convolutions.</summary>
    public int ConvCount { get; }

    /// <summary>Feature channels.</summary>
    public int MelBins { get; }

    /// <summary>Rows one chunk takes in the feature map.</summary>
    public int PaddedFrames { get; }

    /// <summary>Rows one chunk takes after the stem.</summary>
    public int RowsPerChunk => PaddedFrames >> ConvCount;

    /// <summary>The tower's output rows of <paramref name="frames"/> frames (the number of audio placeholder tokens).</summary>
    /// <param name="frames">Valid frames.</param>
    public int OutputRows(int frames)
    {
        int full = frames / ChunkFrames;
        int rest = frames % ChunkFrames;
        return (full * _validPerLevel[ConvCount]) + ValidRows(rest);
    }

    /// <summary>The number of chunks of <paramref name="frames"/> frames.</summary>
    /// <param name="frames">Valid frames.</param>
    public int Chunks(int frames) => (frames + ChunkFrames - 1) / ChunkFrames;

    /// <summary>
    /// The time-major feature map [chunks · <see cref="PaddedFrames"/>, mel bins] of frames <c>[first, first + frames)</c>: the frames of a
    /// partial last chunk are followed by the extractor's next frames up to the chunk's end (the padding of the recording), then zeros.
    /// </summary>
    /// <param name="features">The extracted features (channel-major).</param>
    /// <param name="first">The first frame of the window.</param>
    /// <param name="frames">Valid frames of the window.</param>
    public float[] FeatureMap(AudioFeatures features, int first, int frames)
    {
        int chunks = Chunks(frames);
        var map = new float[(long)chunks * PaddedFrames * MelBins];
        var values = features.Values;
        for (int c = 0; c < chunks; c++)
        {
            for (int t = 0; t < ChunkFrames; t++)
            {
                int frame = first + (c * ChunkFrames) + t;
                if (frame >= features.Frames)
                {
                    break;
                }

                int row = (c * PaddedFrames) + t;
                for (int m = 0; m < MelBins; m++)
                {
                    map[((long)row * MelBins) + m] = values[((long)m * features.Frames) + frame];
                }
            }
        }

        return map;
    }

    /// <summary>The row mask after convolution <paramref name="level"/> (1-based, all but the last): 1 for a chunk's valid rows, 0 for its padding.</summary>
    /// <param name="level">The convolution (1 to <see cref="ConvCount"/> − 1).</param>
    /// <param name="chunks">The number of chunks.</param>
    public float[] RowMask(int level, int chunks)
    {
        int perChunk = PaddedFrames >> level;
        var mask = new float[chunks * perChunk];
        for (int i = 0; i < mask.Length; i++)
        {
            mask[i] = i % perChunk < _validPerLevel[level] ? 1f : 0f;
        }

        return mask;
    }

    /// <summary>The position of every row after the stem within its chunk.</summary>
    /// <param name="chunks">The number of chunks.</param>
    public int[] Positions(int chunks)
    {
        var positions = new int[chunks * RowsPerChunk];
        for (int i = 0; i < positions.Length; i++)
        {
            positions[i] = i % RowsPerChunk;
        }

        return positions;
    }

    /// <summary>The indices of the rows after the stem that hold valid frames, chunk by chunk.</summary>
    /// <param name="frames">Valid frames of the window.</param>
    public int[] ValidRowIndices(int frames)
    {
        int chunks = Chunks(frames);
        var rows = new List<int>(OutputRows(frames));
        for (int c = 0; c < chunks; c++)
        {
            int length = Math.Min(ChunkFrames, frames - (c * ChunkFrames));
            int valid = ValidRows(length);
            for (int r = 0; r < valid; r++)
            {
                rows.Add((c * RowsPerChunk) + r);
            }
        }

        return [.. rows];
    }

    /// <summary>
    /// The attention span (first row, count) of every valid row: consecutive windows of the rows of <c>windowFrames / chunkFrames</c> chunks
    /// (of the longest chunk's row count), the last window holding the rest.
    /// </summary>
    /// <param name="frames">Valid frames of the window.</param>
    public int[] WindowSpans(int frames)
    {
        int rows = OutputRows(frames);
        int longest = frames >= ChunkFrames ? _validPerLevel[ConvCount] : ValidRows(frames);
        int window = Math.Max(1, longest * (WindowFrames / ChunkFrames));
        var spans = new int[2 * rows];
        for (int r = 0; r < rows; r++)
        {
            int start = r / window * window;
            spans[2 * r] = start;
            spans[(2 * r) + 1] = Math.Min(window, rows - start);
        }

        return spans;
    }

    private static int Subsample(int length) => length > 0 ? ((length - 1) / 2) + 1 : 0;

    private int ValidRows(int frames)
    {
        for (int i = 0; i < ConvCount; i++)
        {
            frames = Subsample(frames);
        }

        return frames;
    }

    private bool Fits(int padded)
    {
        for (int level = 0; level < ConvCount; level++)
        {
            if ((padded >> level) < _validPerLevel[level] + 1)
            {
                return false;
            }
        }

        return (padded >> ConvCount) >= _validPerLevel[ConvCount];
    }
}
