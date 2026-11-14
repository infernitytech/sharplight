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
    /// <summary>
    /// The target token of every token of a draft head's own (smaller) vocabulary (int[]; absent: the head's logits are over the target's
    /// vocabulary). EAGLE-3 drafters predict over such a vocabulary.
    /// </summary>
    public const string DraftTokenMap = "draft_token_map";
    /// <summary>
    /// Name of the decode-graph output a model's <see cref="DraftGraph"/> reads as its hidden states (string property; absent:
    /// <see cref="HiddenOutput"/>, the state after the final norm, which the MTP heads of Qwen3.5 / Qwen3-Next and the NextN heads
    /// of DeepSeek-V3 / GLM-4.5 read). For a head trained on another tap of the decoder (e.g. an EAGLE-style head on a
    /// pre-norm or intermediate state), the adapter adds that output to the decode graph and names it here.
    /// </summary>
    public const string DraftHiddenSource = "draft_hidden_source";
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

    /// <summary>
    /// Encoder pass of an encoder-decoder model (T5, BART): input token ids [T] (the whole source sequence, bidirectional attention)
    /// and, when the model has learned positions, positions [T, 1]; output <see cref="EncoderOutput"/> [T, H]. Its output feeds the
    /// <see cref="InputRole.EncoderOutput"/> input [<see cref="EncoderSymbol"/>, H] of the model's <see cref="DecodeGraph"/>, whose
    /// cross-attention reads it on every step.
    /// </summary>
    public const string EncodeGraph = "encode";
    /// <summary>Name of the output of <see cref="EncodeGraph"/> (the encoded source sequence, one row per source token).</summary>
    public const string EncoderOutput = "encoder_output";
    /// <summary>Symbol of the source-sequence length (rows of the <see cref="InputRole.EncoderOutput"/> input).</summary>
    public const string EncoderSymbol = "E";
    /// <summary>Property of encoder-decoder models: the token the decoder starts from (int; T5: the pad token).</summary>
    public const string DecoderStartToken = "decoder_start_token";
    /// <summary>
    /// Masked-diffusion language models (Dream, LLaDA): one bidirectional pass over a single sequence with the inputs of
    /// <see cref="EmbedGraph"/>; output <see cref="LogitsOutput"/> [T, vocabulary], the prediction of every position (see
    /// <see cref="DiffusionShiftLogits"/>). The generation loop starts from the prompt followed by mask tokens
    /// (<see cref="CanonicalKeys.TokenizerMaskId"/>) and reveals them over several passes.
    /// </summary>
    public const string DenoiseGraph = "denoise";
    /// <summary>
    /// Property of diffusion models (bool): row i of the <see cref="DenoiseGraph"/> logits predicts position i + 1 (Dream: trained like an
    /// autoregressive model); false: it predicts position i (LLaDA).
    /// </summary>
    public const string DiffusionShiftLogits = "diffusion.shift_logits";
}
