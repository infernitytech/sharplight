namespace Sharplight.Models;

/// <summary>Semantic role of a graph input so generic runtimes (LLM sessions, vision pipelines) can feed it.</summary>
public enum InputRole
{
    /// <summary>Token ids [T], i32.</summary>
    TokenIds,
    /// <summary>Position ids, shape [axes, T].</summary>
    Positions,
    /// <summary>Precomputed embeddings injected instead of token embeddings (multimodal).</summary>
    Embeddings,
    /// <summary>Indices of rows for which outputs (logits) are requested.</summary>
    OutputRows,
    /// <summary>Image / audio pixel or feature data.</summary>
    Media,
    /// <summary>An input without a runtime-defined meaning; the host feeds it by name.</summary>
    Other,
    /// <summary>
    /// Per-row selector [T, 1] f32 of a graph that takes both token ids and embeddings: 1 = take the row from the
    /// embeddings input, 0 = from the token embedding table (multimodal prompts mixing text and media).
    /// Declared last so the stored values of the other roles (saved packages) stay valid.
    /// </summary>
    EmbeddingMask,
    /// <summary>
    /// Multi-sequence graphs (<see cref="Graph.MultiSequence"/>): per-row table [T, 4] i32 of (slot, position in the slot's
    /// context, row count of the row's run, run index).
    /// </summary>
    SequenceRows,
    /// <summary>Multi-sequence graphs: per-run table [U, 4] i32 of (slot, first row, row count, first position); a run is a
    /// slot's consecutive rows, in context order.</summary>
    SequenceRuns,
    /// <summary>Token type (segment) ids [T] i32 of encoders with a token type embedding table (BERT: 0 = first text, 1 = second).</summary>
    TokenTypes,
    /// <summary>
    /// Packed-sequence graphs (several independent sequences concatenated along the token axis, e.g. <see cref="WellKnown.EmbedGraph"/>):
    /// per-row table [T, 2] i32 of (first row, row count) of the sequence the row belongs to. Attention only looks within it.
    /// </summary>
    RowSpans,
    /// <summary>Packed-sequence graphs: per-sequence table [B, 2] i32 of (first row, row count); rows of one sequence are consecutive.</summary>
    SequenceSpans,
    /// <summary>Per-row sampling parameters [R, 8] i32 of a <see cref="Graph.StandardOps.Sample"/> node (see <see cref="Graph.SamplingRow"/>).</summary>
    SamplingParameters,
    /// <summary>
    /// Paged multi-sequence graphs (<see cref="Graph.MultiSequence"/> with a block size): per-slot block table [S, B] i32; entry
    /// (s, j) is the physical cache block holding positions [j · block size, (j + 1) · block size) of slot s.
    /// </summary>
    KvBlockTable,
    /// <summary>
    /// Multi-sequence graphs with recurrent-state rollback (<see cref="Graph.MultiSequence"/> with rollback rows, see
    /// <see cref="Graph.RecurrentRollback"/>): per-run table [U, 4] i32 of (rows to replay from the slot's rollback records,
    /// record bank to replay from, tentative flag, 0).
    /// </summary>
    SequenceCommits,
    /// <summary>
    /// Hidden states [T, H] f32 of a target decoder fed to a draft head (<see cref="WellKnown.DraftGraph"/>), one row per token.
    /// </summary>
    HiddenStates,
    /// <summary>
    /// Encoder-decoder models: the encoded source sequence [<see cref="WellKnown.EncoderSymbol"/>, H] f32 (the
    /// <see cref="WellKnown.EncoderOutput"/> of the model's <see cref="WellKnown.EncodeGraph"/>) that the decoder's cross-attention reads.
    /// </summary>
    EncoderOutput,
}
