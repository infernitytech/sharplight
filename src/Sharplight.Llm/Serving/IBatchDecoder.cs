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
    /// <summary>
    /// Shortens a slot's context to its first <paramref name="length"/> tokens (prefix reuse). A paged cache keeps less when
    /// <paramref name="length"/> falls inside a block shared with another slot (it never writes into shared blocks): the
    /// context then ends at that block's start; read <see cref="Length"/> afterwards.
    /// </summary>
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

    // ---- paged KV cache (all members are trivial for a decoder whose slots own fixed cache ranges) ----------------------

    /// <summary>Tokens per KV block of a paged cache, 0 when every slot owns a fixed range of <see cref="ContextLength"/> rows.</summary>
    public int KvBlockSize => 0;
    /// <summary>Blocks of the paged cache (0 when not paged).</summary>
    public int KvBlocksTotal => 0;
    /// <summary>Blocks not held by any slot (<see cref="int.MaxValue"/> when not paged: slots never run out of cache).</summary>
    public int KvBlocksFree => int.MaxValue;
    /// <summary>Blocks a slot must newly take to append <paramref name="tokens"/> tokens (0 when not paged).</summary>
    public int KvBlocksNeeded(int slot, int tokens) => 0;
    /// <summary>Blocks that become free when the slot is reset (those no other slot shares; 0 when not paged).</summary>
    public int KvBlocksReleasable(int slot) => 0;
    /// <summary>
    /// Makes an empty slot start with the first <paramref name="tokens"/> tokens of <paramref name="source"/>'s context by sharing
    /// its full blocks (no computation, no copy). Returns the tokens shared: <paramref name="tokens"/> rounded down to whole blocks,
    /// and 0 when the cache is not paged. The slot's <see cref="Length"/> becomes that count.
    /// </summary>
    public int ShareKvPrefix(int slot, int source, int tokens) => 0;
}
