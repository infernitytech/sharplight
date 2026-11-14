namespace Sharplight.Llm.Serving.Speculative;

/// <summary>
/// Rows of one sequence that an <see cref="MtpSpeculator"/> has observed but not yet fed to its draft head: consecutive
/// positions from <see cref="Start"/>, each with the target's hidden state and the token that follows the position.
/// </summary>
internal sealed class MtpRowBuffer(int hiddenSize)
{
    private int[] _tokens = new int[16];
    private float[] _hidden = new float[16 * hiddenSize];

    /// <summary>Position of the first buffered row.</summary>
    public int Start { get; private set; }

    /// <summary>Buffered rows.</summary>
    public int Count { get; private set; }

    /// <summary>Position after the last buffered row.</summary>
    public int End => Start + Count;

    /// <summary>The successor tokens of the buffered rows.</summary>
    public ReadOnlySpan<int> Tokens => _tokens.AsSpan(0, Count);

    /// <summary>The hidden states of the buffered rows (row-major).</summary>
    public ReadOnlySpan<float> Hidden => _hidden.AsSpan(0, Count * hiddenSize);

    /// <summary>Drops all rows; the next append starts at its own position.</summary>
    public void Clear() => Count = 0;

    /// <summary>
    /// Appends rows at <paramref name="position"/>: rows from that position on are replaced; a gap (or a position before the
    /// buffer) restarts the buffer there.
    /// </summary>
    public void Append(int position, ReadOnlySpan<int> successors, ReadOnlySpan<float> hidden)
    {
        if (Count > 0 && position >= Start && position <= End)
        {
            Count = position - Start;
        }
        else
        {
            Count = 0;
        }

        if (Count == 0)
        {
            Start = position;
        }

        int n = successors.Length;
        if (Count + n > _tokens.Length)
        {
            int cap = Math.Max(Count + n, _tokens.Length * 2);
            Array.Resize(ref _tokens, cap);
            Array.Resize(ref _hidden, cap * hiddenSize);
        }

        successors.CopyTo(_tokens.AsSpan(Count));
        hidden[..(n * hiddenSize)].CopyTo(_hidden.AsSpan(Count * hiddenSize));
        Count += n;
    }

    /// <summary>Removes the first <paramref name="n"/> rows (they were fed to the draft head).</summary>
    public void RemoveFirst(int n)
    {
        _tokens.AsSpan(n, Count - n).CopyTo(_tokens);
        _hidden.AsSpan(n * hiddenSize, (Count - n) * hiddenSize).CopyTo(_hidden);
        Start += n;
        Count -= n;
    }
}
