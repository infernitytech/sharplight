namespace Sharplight.Llm.Serving;

/// <summary>Rows one sequence contributes to a <see cref="IBatchDecoder.Step"/>.</summary>
/// <param name="Slot">The sequence slot (0 .. <see cref="IBatchDecoder.SlotCount"/> - 1) whose context the rows extend.</param>
/// <param name="Tokens">Token ids appended to the slot's context (prompt chunk or generated token).</param>
/// <param name="WantLogits">Whether the logits of the last row are needed (false for non-final prompt chunks).</param>
public readonly record struct SequenceInput(int Slot, ReadOnlyMemory<int> Tokens, bool WantLogits = true);

/// <summary>
/// The model-execution contract of the serving layer: a decoder holding <see cref="SlotCount"/> independent sequence
/// contexts (KV cache / recurrent state per slot) that advances any subset of them in one forward pass.
/// Implementations: <see cref="SessionBatchDecoder"/> (one slot over an <see cref="InferenceSession"/>, inputs run one
/// after another) and the multi-sequence decoder of the runtime, which batches rows of different slots together.
/// Not thread-safe: one scheduler thread drives it.
/// </summary>
public interface IBatchDecoder : IDisposable
{
    public int SlotCount { get; }
    /// <summary>Maximum context length per slot.</summary>
    public int ContextLength { get; }
    /// <summary>Maximum rows (summed over all inputs) per <see cref="Step"/>.</summary>
    public int MaxBatchTokens { get; }
    public int VocabSize { get; }
    /// <summary>True when one <see cref="Step"/> really executes rows of several slots in one pass.</summary>
    public bool BatchesSequences { get; }
    /// <summary>Tokens currently in a slot's context.</summary>
    public int Length(int slot);
    /// <summary>Empties a slot.</summary>
    public void Reset(int slot);
    /// <summary>
    /// Whether <see cref="Truncate"/> works: true for pure-attention models (dropping KV rows), false when a slot holds
    /// recurrent state that cannot be rewound (then prefix reuse is limited to whole-context matches).
    /// </summary>
    public bool CanTruncate { get; }
    /// <summary>Shortens a slot's context to its first <paramref name="length"/> tokens (prefix reuse).</summary>
    public void Truncate(int slot, int length);
    /// <summary>
    /// Runs one forward pass over the given inputs (at most one per slot, total rows ≤ <see cref="MaxBatchTokens"/>).
    /// Returns one logits array (length <see cref="VocabSize"/>) per input with <see cref="SequenceInput.WantLogits"/>,
    /// in input order; null entries for the others. The arrays may be reused by the next call.
    /// </summary>
    public IReadOnlyList<float[]?> Step(IReadOnlyList<SequenceInput> inputs);
    /// <summary>Accelerator (GPU) memory held by this decoder's compiled model: weights, state, activations (0 when unknown or all on the CPU).</summary>
    public long DeviceMemoryBytes => 0;
}

/// <summary>Single-slot <see cref="IBatchDecoder"/> over an <see cref="InferenceSession"/>.</summary>
public sealed class SessionBatchDecoder(InferenceSession session, bool ownsSession = true) : IBatchDecoder
{
    private readonly float[] _logits = new float[session.VocabSize];

    public int SlotCount => 1;
    public int ContextLength => session.ContextLength;
    public int MaxBatchTokens => (int)session.Compiled.Options.MaxTokens;
    public int VocabSize => session.VocabSize;
    public bool BatchesSequences => false;
    public bool CanTruncate => false;
    public long DeviceMemoryBytes => session.Compiled.AcceleratorMemoryBytes;
    public int Length(int slot) { Check(slot); return session.Position; }
    public void Reset(int slot) { Check(slot); session.Reset(); }
    public void Truncate(int slot, int length)
    {
        Check(slot);
        if (length == session.Position)
        {
            return;
        }

        if (length != 0)
        {
            throw new NotSupportedException("This decoder cannot truncate a context.");
        }

        session.Reset();
    }

    public IReadOnlyList<float[]?> Step(IReadOnlyList<SequenceInput> inputs)
    {
        var res = new float[]?[inputs.Count];
        for (int i = 0; i < inputs.Count; i++)
        {
            Check(inputs[i].Slot);
            var lg = session.Process(inputs[i].Tokens.Span);
            if (inputs[i].WantLogits) { lg.CopyTo(_logits, 0); res[i] = _logits; }
        }
        return res;
    }

    private static void Check(int slot) { if (slot != 0)
        {
            throw new ArgumentOutOfRangeException(nameof(slot));
        }
    }

    public void Dispose() { if (ownsSession)
        {
            session.Dispose();
        }
    }
}
