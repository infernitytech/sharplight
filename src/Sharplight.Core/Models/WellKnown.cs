namespace Sharplight.Models;

/// <summary>Well-known graph and property names shared by adapters and runtimes.</summary>
public static class WellKnown
{
    /// <summary>Name of the autoregressive decoder graph: token ids in, logits out, advancing the KV cache.</summary>
    public const string DecodeGraph = "decode";
    /// <summary>Name of the graph that turns preprocessed image patches into embeddings for the decoder.</summary>
    public const string VisionEncodeGraph = "vision_encode";
    /// <summary>
    /// Image-encoder graph of the vision task layer (ImageEncoderAdapter): input <c>patches</c> f32 [N, in_channels·patch²] (the normalized image cut
    /// into patches in raster order, each flattened channel-major), outputs <see cref="PooledOutput"/> f32 [1, hidden] (pooled, normed
    /// image embedding) and optionally <see cref="LogitsOutput"/> f32 [1, labels] (classifier head).
    /// </summary>
    public const string ImageEncodeGraph = "image_encode";
    /// <summary>
    /// The text tower of a dual-encoder (image + text) model: tokens i32 [L] (L fixed by the component, padded by the host) and <c>pool_index</c> i32 [1]
    /// (the row to pool) in; <see cref="PooledOutput"/> f32 [1, dim] (the text embedding in the image embedding space) and optionally the zero-shot
    /// <c>logit_scale</c> and <c>logit_bias</c> [1] out.
    /// </summary>
    public const string TextEncodeGraph = "text_encode";
    /// <summary>
    /// A set-prediction object detector: <c>pixels</c> f32 [1, 3, H, W] (the normalized image) in; <see cref="LogitsOutput"/> f32 [Q, C] (per query
    /// class scores) and <see cref="BoxesOutput"/> f32 [Q, 4] (center x, center y, width, height as fractions of the model input) out.
    /// </summary>
    public const string DetectGraph = "detect";
    /// <summary>Name of the box output of <see cref="DetectGraph"/>.</summary>
    public const string BoxesOutput = "boxes";
    /// <summary>
    /// Dense depth estimation: <c>patches</c> f32 [N, 3 · patch²] (the host-patchified normalized image, as <see cref="ImageEncodeGraph"/>) in;
    /// <see cref="DepthOutput"/> f32 [1, 1, H, W] (relative inverse depth at the model input resolution) out.
    /// </summary>
    public const string DepthGraph = "estimate_depth";
    /// <summary>Name of the depth map output of <see cref="DepthGraph"/>.</summary>
    public const string DepthOutput = "depth";
    /// <summary>
    /// Per-pixel (semantic) segmentation: <c>pixels</c> f32 [1, 3, H, W] (the normalized image) in; <see cref="SemanticLogitsOutput"/> f32 [1, K, h, w] (class scores
    /// per pixel at the model's output resolution, often a fraction of the input) out.
    /// </summary>
    public const string SemanticSegmentGraph = "segment_semantic";
    /// <summary>Name of the class score output of <see cref="SemanticSegmentGraph"/>.</summary>
    public const string SemanticLogitsOutput = "semantic_logits";
    /// <summary>
    /// Promptable segmentation, image stage: <c>pixels</c> f32 [1, 3, H, W] (the normalized image) in; the image embedding <see cref="SegmentEmbeddingOutput"/>
    /// [h·w, C] (rows in raster order) and the two high-resolution feature maps <see cref="SegmentFeatures0Output"/> and <see cref="SegmentFeatures1Output"/> ([1, C, H, W]) out. Run once per image.
    /// </summary>
    public const string SegmentEncodeGraph = "segment_encode";
    /// <summary>
    /// Promptable segmentation, prompt stage: the three outputs of <see cref="SegmentEncodeGraph"/> (inputs of the same names), <c>point_coords</c> f32 [P, 2]
    /// (x, y in pixels of the model input) and <c>point_kinds</c> i32 [P] (<c>Sharplight.Vision.SegmentPromptKinds</c>) in; <see cref="SegmentMasksOutput"/> f32 [M, h', w']
    /// (mask logits at the model's mask resolution: row 0 the single-mask candidate, the others the multi-mask candidates), <see cref="SegmentScoresOutput"/> f32 [M]
    /// (predicted IoU per candidate) and <see cref="SegmentObjectOutput"/> f32 [1] (object presence logit) out. P is the symbol <see cref="SegmentPointSymbol"/>.
    /// </summary>
    public const string SegmentDecodeGraph = "segment_decode";
    /// <summary>Name of the image embedding output of <see cref="SegmentEncodeGraph"/>.</summary>
    public const string SegmentEmbeddingOutput = "embedding";
    /// <summary>Name of the highest-resolution feature output of <see cref="SegmentEncodeGraph"/>.</summary>
    public const string SegmentFeatures0Output = "features_0";
    /// <summary>Name of the second feature output of <see cref="SegmentEncodeGraph"/>.</summary>
    public const string SegmentFeatures1Output = "features_1";
    /// <summary>Name of the mask logits output of <see cref="SegmentDecodeGraph"/>.</summary>
    public const string SegmentMasksOutput = "masks";
    /// <summary>Name of the IoU score output of <see cref="SegmentDecodeGraph"/>.</summary>
    public const string SegmentScoresOutput = "scores";
    /// <summary>Name of the object presence output of <see cref="SegmentDecodeGraph"/>.</summary>
    public const string SegmentObjectOutput = "object_score";
    /// <summary>Symbol of the prompt point count of <see cref="SegmentDecodeGraph"/>.</summary>
    public const string SegmentPointSymbol = "P";
    /// <summary>
    /// Image upscaling: <c>pixels</c> f32 [1, 3, H, W] (RGB in 0..1) in; <see cref="UpscaledOutput"/> f32 [1, 3, H · s, W · s] (RGB, about 0..1) out. The size is fixed
    /// per graph (the <see cref="UpscaleHeightDimension"/> and <see cref="UpscaleWidthDimension"/> of <see cref="AdapterOptions.Dimensions"/> rebuild it for other sizes).
    /// </summary>
    public const string UpscaleGraph = "upscale";
    /// <summary>Name of the image output of <see cref="UpscaleGraph"/>.</summary>
    public const string UpscaledOutput = "upscaled";
    /// <summary>Dimension name of the input height of <see cref="UpscaleGraph"/>.</summary>
    public const string UpscaleHeightDimension = "height";
    /// <summary>Dimension name of the input width of <see cref="UpscaleGraph"/>.</summary>
    public const string UpscaleWidthDimension = "width";
    /// <summary>Name of the pooled embedding output of <see cref="ImageEncodeGraph"/>.</summary>
    public const string PooledOutput = "pooled";
    /// <summary>Decoder taking token ids, an embeddings input and a per-row mask (<see cref="InputRole.EmbeddingMask"/>): text mixed with media embeddings.</summary>
    public const string DecodeMultimodalGraph = "decode_multimodal";
    /// <summary>Symbol of the token (row) count of a graph execution.</summary>
    public const string TokenSymbol = "T";
    /// <summary>Symbol of the context length that sizes KV caches.</summary>
    public const string ContextSymbol = "ctx";
    /// <summary>Runtime scalar: number of tokens already in the KV cache before this execution.</summary>
    public const string PastScalar = "n_past";
    /// <summary>
    /// Runtime scalar: the first row of the per-source states (cached cross-attention keys and values) that the current pass writes; 0
    /// for models that write the whole source at once.
    /// </summary>
    public const string SourcePastScalar = "source_past";
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
    /// <summary>
    /// The <see cref="DraftGraph"/> of a model with a vision or audio tower, with the media inputs of <see cref="DecodeMultimodalGraph"/>
    /// (<see cref="InputRole.Embeddings"/> and <see cref="InputRole.EmbeddingMask"/>): a row whose next row is an image or audio row combines
    /// that row's embedding, not its placeholder token's, with the target's hidden state.
    /// </summary>
    public const string DraftMultimodalGraph = "mtp_draft_multimodal";
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
    /// <summary>
    /// Whether a <see cref="DraftGraph"/> drafts every token of a step from the same position (bool property; absent: false). Such a head
    /// (Gemma 4 assistants: the queries of the pending token's position against the target's keys and values) keeps one row per verified target
    /// row; a further draft replaces the head's last row instead of adding one, with that row's hidden-state input except for the first
    /// <see cref="HiddenOutput"/>-wide part, which carries the head's previous output.
    /// </summary>
    public const string DraftAnchored = "draft_anchored";
    /// <summary>
    /// Block drafting (int property; absent or 0: the <see cref="DraftGraph"/> drafts one token per pass). The draft graph denoises a block of
    /// the pending token and <see cref="DraftMaskToken"/> tokens in one bidirectional pass, after rows that carry the target's
    /// <see cref="DraftHiddenSource"/> features of the positions before it (DFlash drafters); the value is the block length it was trained on
    /// (at most one draft per mask row). Its hidden-state input rows are [features | context flag | per convolution tap: previous row, valid |
    /// with a candidate lattice: predecessor logits row, predecessor is the pending token] (see <see cref="DraftConvTaps"/> and
    /// <see cref="DraftLatticeCandidates"/>).
    /// </summary>
    public const string DraftBlockSize = "draft_block_size";
    /// <summary>The token of the block rows after the pending token of a block drafter (int property, <see cref="DraftBlockSize"/>).</summary>
    public const string DraftMaskToken = "draft_mask_token";
    /// <summary>
    /// Earlier rows of the same block a block drafter's per-row convolution reads (int property; absent: 0). Each takes two columns of the
    /// hidden-state input: the step row it reads and whether that row lies in the block.
    /// </summary>
    public const string DraftConvTaps = "draft_conv_taps";
    /// <summary>
    /// Candidates per row of a block drafter whose logits output is a candidate lattice (int property k; absent or 0: the logits output is the
    /// greedy token of every row, one column). A lattice row holds k candidate tokens and k·k scores, score[p·k + c] of candidate c after
    /// candidate p of the previous row (after the pending token for the first mask row); drafts follow the best successor of the previous choice.
    /// </summary>
    public const string DraftLatticeCandidates = "draft_lattice_candidates";
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
    /// <summary>
    /// Property: with several position axes, how many leading axes carry a text token's position; the others are 0 (int, default
    /// <see cref="DefaultTextPositionAxes"/>: time, row and column of M-RoPE). Models whose extra axis rotates text tokens too (XD-RoPE) set it
    /// to the axis count.
    /// </summary>
    public const string TextPositionAxes = "text_position_axes";
    /// <summary>Default of <see cref="TextPositionAxes"/>.</summary>
    public const int DefaultTextPositionAxes = 3;
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

    /// <summary>Output of an image-and-prompt encoder graph: the rows the vision tower and projector produce, before they are merged with the prompt.</summary>
    public const string ImageFeaturesOutput = "image_features";
    /// <summary>Symbol of the source-sequence length (rows of the <see cref="InputRole.EncoderOutput"/> input).</summary>
    public const string EncoderSymbol = "E";
    /// <summary>
    /// Symbol of the frame count of a variable-length audio encoder's <see cref="InputRole.AudioFeatures"/> input [channels, frames] (samples for
    /// encoders over the raw waveform); fixed-window encoders (Whisper) have a fixed frame count instead.
    /// </summary>
    public const string AudioFrameSymbol = "F";
    /// <summary>
    /// Prediction network of a transducer speech recognizer (RNN-T, TDT): input token ids [1] (<see cref="InputRole.TokenIds"/>) and the
    /// recurrent state inputs <see cref="TransducerHiddenState"/> and <see cref="TransducerCellState"/> [layers, J]; outputs
    /// <see cref="TransducerPrediction"/> [1, J] and the advanced states under the same names. The host keeps the states, so a hypothesis
    /// can be continued or abandoned without rolling the graph back.
    /// </summary>
    public const string TransducerPredictGraph = "predict";
    /// <summary>
    /// Joint network of a transducer speech recognizer: inputs one projected encoder row [1, J] (<see cref="InputRole.EncoderOutput"/>) and
    /// <see cref="TransducerPrediction"/> [1, J] (<see cref="InputRole.HiddenStates"/>); output <see cref="LogitsOutput"/> [1, tokens + durations].
    /// </summary>
    public const string TransducerJointGraph = "joint";
    /// <summary>Hidden-state input and output of <see cref="TransducerPredictGraph"/>.</summary>
    public const string TransducerHiddenState = "state_h";
    /// <summary>Cell-state input and output of <see cref="TransducerPredictGraph"/>.</summary>
    public const string TransducerCellState = "state_c";
    /// <summary>The prediction output of <see cref="TransducerPredictGraph"/> (the input of <see cref="TransducerJointGraph"/>).</summary>
    public const string TransducerPrediction = "prediction";
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
    /// <summary>
    /// Span scoring (decision models: a prompt holds questions and their options, the head gives every option a score): the packed-sequence
    /// inputs of <see cref="EmbedGraph"/> plus <see cref="InputRole.QuestionSpans"/> [<see cref="QuestionSymbol"/>, 6] and
    /// <see cref="InputRole.OptionSpans"/> [<see cref="OptionSymbol"/>, 3], which mark spans of the packed rows; output
    /// <see cref="ScoresOutput"/> [<see cref="OptionSymbol"/>, 1], one score per option (the options of a question are compared by a softmax
    /// over their scores). How the head reads the spans is the model's (<see cref="SpanHeadType"/>). Recurrent and hybrid models whose layers
    /// keep state instead have the inputs of <see cref="DecodeGraph"/> (tokens, positions, the <see cref="PastScalar"/> offset and their
    /// states) plus <see cref="InputRole.SequenceSpans"/> and the span tables: one sequence per run, started at offset 0 from reset states.
    /// </summary>
    public const string ScoreSpansGraph = "score_spans";
    /// <summary>Name of the output of <see cref="ScoreSpansGraph"/>: one score per option.</summary>
    public const string ScoresOutput = "scores";
    /// <summary>Symbol of the question count of <see cref="ScoreSpansGraph"/>.</summary>
    public const string QuestionSymbol = "Q";
    /// <summary>Symbol of the option count of <see cref="ScoreSpansGraph"/>.</summary>
    public const string OptionSymbol = "O";
    /// <summary>
    /// Property of models with a <see cref="ScoreSpansGraph"/> (string): the head type, e.g. "endpoint" (the last token of each option against
    /// the last token of its question) or "joint_schema" (pooled spans read by cross-attention layers, all questions decided jointly).
    /// </summary>
    public const string SpanHeadType = "span_head_type";
}
