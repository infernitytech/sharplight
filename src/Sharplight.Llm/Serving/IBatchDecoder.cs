namespace Sharplight.Llm.Serving;

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
