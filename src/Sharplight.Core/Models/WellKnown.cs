namespace Sharplight.Models;

/// <summary>Well-known graph and property names shared by adapters and runtimes.</summary>
public static class WellKnown
{
    /// <summary>Name of the autoregressive decoder graph: token ids in, logits out, advancing the KV cache.</summary>
    public const string DecodeGraph = "decode";
    /// <summary>Name of the graph that turns preprocessed image patches into embeddings for the decoder.</summary>
    public const string VisionEncodeGraph = "vision_encode";
    /// <summary>Decoder taking token ids, an embeddings input and a per-row mask (<see cref="InputRole.EmbeddingMask"/>): text mixed with media embeddings.</summary>
    public const string DecodeMultimodalGraph = "decode_multimodal";
    /// <summary>Symbol of the token (row) count of a graph execution.</summary>
    public const string TokenSymbol = "T";
    /// <summary>Symbol of the context length that sizes KV caches.</summary>
    public const string ContextSymbol = "ctx";
    /// <summary>Runtime scalar: number of tokens already in the KV cache before this execution.</summary>
    public const string PastScalar = "n_past";
    /// <summary>Symbol of the run count of a multi-sequence graph (rows of <see cref="InputRole.SequenceRuns"/>).</summary>
    public const string RunSymbol = "U";

    /// <summary>
    /// Packed-sequence graph that turns a batch of sequences into one vector each (embedding models): inputs token ids [T],
    /// positions [T, 1] (each sequence starting at 0), optional token types [T], <see cref="InputRole.RowSpans"/> and
    /// <see cref="InputRole.SequenceSpans"/> [B, 2]; output <see cref="EmbeddingsOutput"/> [B, H] (pooled as the model's
    /// <see cref="EmbeddingPooling"/> property says and L2-normalized when <see cref="EmbeddingNormalize"/> is set) — or, for
    /// pooling "none", [T, H] (one vector per token).
    /// </summary>
    public const string EmbedGraph = "embed";
    /// <summary>
    /// Packed-sequence graph with the same inputs as <see cref="EmbedGraph"/> that scores each sequence with a head:
    /// output <see cref="LogitsOutput"/> [B, C] (cross-encoder classifier: C labels; decoder: the language model head over the
    /// sequence's last row, C = vocabulary).
    /// </summary>
    public const string ClassifyGraph = "classify";
    /// <summary>Name of the embedding graph output (pooled vectors).</summary>
    public const string EmbeddingsOutput = "embeddings";
    /// <summary>Name of the logits output of the decoder and classifier graphs.</summary>
    public const string LogitsOutput = "logits";
    /// <summary>
    /// Name of the hidden-state output [T, H] of the decoder graphs: what the language model head reads (after the final norm),
    /// and what a draft head (<see cref="DraftGraph"/>) takes as its <see cref="InputRole.HiddenStates"/> input.
    /// </summary>
    public const string HiddenOutput = "hidden";
    /// <summary>
    /// Multi-token-prediction draft head (MTP / NextN layers) of a decoder, for speculative decoding: inputs token ids [T] (the
    /// token following each row's position), <see cref="InputRole.HiddenStates"/> [T, H] (the target decoder's
    /// <see cref="HiddenOutput"/> at that position, or for chained drafts this graph's own <see cref="HiddenOutput"/>), positions
    /// (the target positions of the hidden states) and output rows; outputs <see cref="LogitsOutput"/> [R, V] (the token two
    /// positions ahead) and <see cref="HiddenOutput"/> [T, H] (fed back for the next draft). Attention-only state (its own KV cache).
    /// </summary>
    public const string DraftGraph = "mtp_draft";
    /// <summary>Number of draft (MTP / NextN) layers a model ships (int; 0 or absent: no <see cref="DraftGraph"/>).</summary>
    public const string DraftLayers = "draft_layers";
    /// <summary>Symbol of the sequence count of a packed-sequence graph (rows of <see cref="InputRole.SequenceSpans"/>).</summary>
    public const string SequenceSymbol = "B";

    /// <summary>Property: vocabulary size (int).</summary>
    public const string VocabSize = "vocab_size";
    /// <summary>Property: maximum context length the model supports (int).</summary>
    public const string ContextLength = "context_length";
    /// <summary>Property: hidden size of the model (int).</summary>
    public const string EmbeddingLength = "embedding_length";
    /// <summary>
    /// Property: number of position axes the positions input carries (1 for plain RoPE, more for multi-axis RoPE) (int).
    /// </summary>
    public const string PositionAxes = "position_axes";
    /// <summary>Property: token ids that end generation (int[]).</summary>
    public const string EosTokenIds = "eos_token_ids";
    /// <summary>Pooling of <see cref="EmbedGraph"/>: "mean", "cls" (first row), "last", "none" (string).</summary>
    public const string EmbeddingPooling = "embedding_pooling";
    /// <summary>Whether <see cref="EmbedGraph"/> L2-normalizes its output (bool).</summary>
    public const string EmbeddingNormalize = "embedding_normalize";
    /// <summary>Whether attention of the packed-sequence graphs is causal (decoder-based embedders) (bool).</summary>
    public const string CausalAttention = "causal_attention";
    /// <summary>Label count of <see cref="ClassifyGraph"/> (int; the vocabulary size for decoders).</summary>
    public const string ClassifierLabels = "classifier_labels";
}
