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
    /// <summary>Number of independent sequence contexts (slots) the decoder holds.</summary>
    public int SlotCount { get; }
    /// <summary>Maximum context length per slot.</summary>
    public int ContextLength { get; }
    /// <summary>Maximum rows (summed over all inputs) per <see cref="Step"/>.</summary>
    public int MaxBatchTokens { get; }
    /// <summary>Length of the logits arrays returned by <see cref="Step"/>.</summary>
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
    /// Most logits rows one <see cref="Step"/> can return in total (the sum of <see cref="SequenceInput.OutputRows"/>); at least
    /// <see cref="SlotCount"/>. Larger values allow speculative verification (several rows of one sequence).
    /// </summary>
    public int MaxLogitsRows => SlotCount;
    /// <summary>
    /// Runs one forward pass over the given inputs (at most one per slot, total rows ≤ <see cref="MaxBatchTokens"/>).
    /// Returns, in input order, the logits of each input: <see cref="SequenceInput.OutputRows"/> consecutive rows of
    /// <see cref="VocabSize"/> floats (row-major, oldest row first), or an empty slice for inputs without logits.
    /// The slices point into decoder-owned memory that the next call overwrites; callers may modify them in place.
    /// </summary>
    public IReadOnlyList<Memory<float>> Step(IReadOnlyList<SequenceInput> inputs);
    /// <summary>Accelerator (GPU) memory held by this decoder's compiled model: weights, state, activations (0 when unknown or all on the CPU).</summary>
    public long DeviceMemoryBytes => 0;
}
