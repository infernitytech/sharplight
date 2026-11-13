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
}
