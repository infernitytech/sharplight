namespace Sharplight.Graph;

/// <summary>
/// Sampling parameters of one logits row of a <see cref="StandardOps.Sample"/> node, encoded as <see cref="Width"/> i32 values
/// (the node's <c>params</c> input). <see cref="Temperature"/> ≤ 0 selects the arg-max (the first maximum on ties). Otherwise the
/// token is drawn from the softmax of <c>logits / Temperature</c> restricted by the filters (see <see cref="SamplingHead"/>), using
/// random numbers that depend only on (<see cref="Seed"/>, <see cref="Counter"/>, token id): the same row always yields the same token.
/// </summary>
public readonly record struct SamplingRow
{
    /// <summary>Number of i32 values one encoded row takes.</summary>
    public const int Width = 8;

    private const int ActiveFlag = 1;
    private const int GreedyFlag = 2;
    private const int LogprobFlag = 4;

    /// <summary>Softmax temperature; 0 or less selects the most likely token.</summary>
    public float Temperature { get; init; }
    /// <summary>Keep only the K most likely tokens (all tokens tied with the K-th are kept); 0 disables.</summary>
    public int TopK { get; init; }
    /// <summary>Nucleus threshold: keep the most likely tokens until their probability mass reaches this value; values outside (0, 1) disable it.</summary>
    public float TopP { get; init; }
    /// <summary>Drop tokens less likely than this fraction of the most likely one; 0 disables.</summary>
    public float MinP { get; init; }
    /// <summary>Random stream of the sequence.</summary>
    public uint Seed { get; init; }
    /// <summary>Position in the sequence's random stream; callers advance it once per sampled token.</summary>
    public uint Counter { get; init; }
    /// <summary>Also compute the log-probability (log-softmax of the raw logits) of the chosen token.</summary>
    public bool WantLogprob { get; init; }

    /// <summary>Arg-max selection.</summary>
    /// <param name="wantLogprob">Also compute the chosen token's log-probability.</param>
    public static SamplingRow Greedy(bool wantLogprob = false) => new() { WantLogprob = wantLogprob };

    /// <summary>True when the row selects the arg-max.</summary>
    public bool IsGreedy => !(Temperature > 0);

    /// <summary>Encodes the row into <paramref name="destination"/> (at least <see cref="Width"/> values).</summary>
    public void WriteTo(Span<int> destination)
    {
        destination[0] = ActiveFlag | (IsGreedy ? GreedyFlag : 0) | (WantLogprob ? LogprobFlag : 0);
        destination[1] = Math.Max(0, TopK);
        destination[2] = BitConverter.SingleToInt32Bits(Temperature);
        destination[3] = BitConverter.SingleToInt32Bits(TopP);
        destination[4] = BitConverter.SingleToInt32Bits(MinP);
        destination[5] = unchecked((int)Seed);
        destination[6] = unchecked((int)Counter);
        destination[7] = 0;
    }

    /// <summary>Encodes a row the op skips (its token output is -1): rows whose logits the caller samples itself.</summary>
    public static void WriteSkip(Span<int> destination) => destination[..Width].Clear();

    /// <summary>Decodes an encoded row; returns false for a skipped row.</summary>
    public static bool TryRead(ReadOnlySpan<int> source, out SamplingRow row)
    {
        int flags = source[0];
        row = new SamplingRow
        {
            Temperature = (flags & GreedyFlag) != 0 ? 0 : BitConverter.Int32BitsToSingle(source[2]),
            TopK = source[1],
            TopP = BitConverter.Int32BitsToSingle(source[3]),
            MinP = BitConverter.Int32BitsToSingle(source[4]),
            Seed = unchecked((uint)source[5]),
            Counter = unchecked((uint)source[6]),
            WantLogprob = (flags & LogprobFlag) != 0,
        };
        return (flags & ActiveFlag) != 0;
    }
}
