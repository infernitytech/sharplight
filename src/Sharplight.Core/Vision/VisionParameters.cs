using System.Globalization;
using Sharplight.Models;

namespace Sharplight.Vision;

/// <summary>
/// Canonical vision-tower vocabulary, shared by every importer, architecture adapter and the image
/// preprocessor. Independent of any file format (llama.cpp "mmproj", Hugging Face <c>vision_config</c>, ONNX, ...).
/// </summary>
/// <remarks>
/// <para><b>Hyper-parameters</b> are stored in a <see cref="ModelSource"/> as <c>{architecture}.vision.{key}</c>
/// (the architecture being the text model the tower belongs to, e.g. <c>qwen3_5.vision.layer_count</c>) and in a built
/// <see cref="ModelDefinition.Properties"/> as <c>vision.{key}</c>; keys:</para>
/// <list type="bullet">
/// <item><c>layer_count</c>, <c>hidden_size</c>, <c>ffn_hidden_size</c>, <c>head_count</c>, <c>norm_eps</c> (ViT trunk);
/// <c>activation</c> (<c>gelu_tanh</c> | <c>gelu</c> | <c>silu</c> | <c>quick_gelu</c> = x·sigmoid(1.702·x); trunk MLP, the gate's activation when the MLP is gated);
/// <c>norm.type</c> (<c>layer</c>, the default: LayerNorm with weight and bias | <c>rms</c>: RMS norm with a weight; every norm of the tower and merger);
/// <c>norm.placement</c> (<c>pre</c>, the default: x + attn(input_norm(x)), x + mlp(post_attention_norm(x)) | <c>post</c>: x + input_norm(attn(x)),
/// x + post_attention_norm(mlp(x)); EVA2-CLIP, CogVLM); <c>class_token</c> (bool: a learned class row, <c>vision.class_embedding</c> plus
/// <c>vision.class_position_embedding</c>, joins the patches in the trunk; it is not part of the output);</item>
/// <item><c>attention.window</c> (side of the square attention windows in merged tokens; 0 = none, the default) with <c>attention.full_layers</c>
/// (int[], ascending layers that attend over the whole image): windowed attention (Qwen2.5-VL). The merged-token grid is cut into windows of
/// window×window merged tokens (the last row/column of windows is smaller when the grid is not a multiple), patches are ordered window by
/// window and a patch of a windowed layer only attends to the patches of its window;</item>
/// <item><c>image_size</c> (side of the square every image is resized to, ignoring its aspect ratio; 0, the default: the aspect-preserving
/// resize within <c>min_pixels</c> / <c>max_pixels</c>);</item>
/// <item><c>patch_size</c>, <c>temporal_patch_size</c> (frames per patch; a still image is repeated), <c>in_channels</c>,
/// <c>spatial_merge_size</c> (m: m×m neighbouring patches are merged into one output token);</item>
/// <item><c>position_embedding_count</c> (learned absolute position grid of g×g entries, g² = count, resampled to the image grid; 0 = none),
/// <c>position_embedding.interpolation</c> (<c>bilinear</c>, the default: align_corners, 4 taps | <c>bilinear_half_pixel</c>: PyTorch's bilinear
/// resize with align_corners off, the patch centres sampled, 4 taps | <c>bicubic</c>: PyTorch grid_sample of the patch centres, 16 taps),
/// <c>rope.base</c> (axial 2-D rotary position embedding over (row, col), 0 = none);</item>
/// <item><c>out_hidden_size</c> (width of the produced embeddings = hidden size of the language model),
/// <c>merger.activation</c> (<c>gelu</c> | <c>gelu_tanh</c> | <c>silu</c>), <c>merger.norm_eps</c> (epsilon of the merger norms; default
/// <c>norm_eps</c>), <c>merger.norm.type</c> (kind of the merger norms; default <c>norm.type</c>);</item>
/// <item><c>merger.row_separator</c> (bool: a learned row is appended after every row of merged tokens, HunyuanVL's newline column) and
/// <c>merger.boundary</c> (bool: learned begin and end rows frame the image); the image then occupies
/// <see cref="OutputTokenCount"/> rows of the language model, in the order begin, each grid row followed by its separator, end;</item>
/// <item><c>image_positions</c>: the language model's M-RoPE positions of an image's rows. <c>mrope</c> (the default, Qwen-VL): row r, column c
/// of an image starting at position p get (p, p + r, p + c, 0) and the text after it continues at p + max(rows, columns); <c>sequential</c>
/// (HunyuanVL's XD-RoPE): every row advances the first axis by one like text, a grid row (separators included, at column = columns) gets
/// (column, row, image ordinal) on the other three axes, the begin and end rows carry their sequential position on every axis; <c>grouped</c>
/// (CogVLM): the begin row, all grid rows together and the end row take one position each, on every axis;</item>
/// <item><c>media_weight_rows</c>: which rows of an image use the language model's media weight set (layers with <c>media.</c> weights, visual
/// experts): <c>all</c> (the default) | <c>all_but_last</c> (CogVLM: the end row uses the text weights);</item>
/// <item><c>image_token</c> (text of the prompt placeholder that expands to the image rows; default <c>&lt;|image_pad|&gt;</c>);</item>
/// <item><c>deepstack_layers</c> (int[], ascending tower layer indices; default none): DeepStack (Qwen3-VL). The output of each listed
/// layer is merged like the final hidden states and projected by its own merger; the results are appended to the embedding row, which is
/// then <see cref="EmbeddingWidth"/> = out_hidden_size·(1 + count) wide. The language model adds the k-th appended block to the hidden
/// state after its k-th layer;</item>
/// <item>Gemma 4 towers: <c>position_embedding.axial</c> (bool: <c>vision.position_embedding.weight</c> is [2·count, hidden], the x table (columns)
/// followed by the y table (rows), summed per patch; <c>position_embedding_count</c> is the rows of one table), <c>rope.layout</c> (<c>paired</c>, the
/// default, or <c>split</c>: the first half of every head rotates by the column, the second by the row, each half in rotate-half order),
/// <c>attention.scale</c> (0, the default, = 1/√head_dim), <c>attention.qk_norm</c> (bool: RMS norms of every head's query and key,
/// <c>attention.q_norm.weight</c> / <c>attention.k_norm.weight</c> [head_dim]), <c>attention.v_norm</c> (bool: an RMS norm without weight on the values),
/// <c>norm.placement</c> <c>sandwich</c> (x + post_norm(attn(input_norm(x))), x + post_norm(mlp(pre_norm(x))): <c>input_norm</c>, <c>attention.post_norm</c>,
/// <c>post_attention_norm</c> and <c>mlp.post_norm</c>), <c>merger.pool</c> (bool: the m×m patches of a merge unit are averaged instead of concatenated and the
/// mean is scaled by √hidden), <c>standardize</c> (bool: <c>vision.std_bias</c> is subtracted from and <c>vision.std_scale</c> multiplies every pooled row),
/// then an RMS norm without weight and <c>vision.merger.out</c> (no bias) project to the language model's width; <c>resize.mode</c> (<c>smart</c>, the default, or
/// <c>fit_budget</c>: the image is scaled so that it just fits max_pixels, sides rounded down to a multiple of patch_size·merge_size, also enlarged),
/// <c>image_begin_token</c> / <c>image_end_token</c> (texts of tokens that frame the image rows of the prompt);</item>
/// <item>Gemma 3 towers: <c>merger.pool</c> with <c>merger.pool.scale</c> <c>mean</c> (the plain average; the default <c>sqrt_hidden</c> is Gemma 4's
/// √hidden-scaled mean), no standardization, then the RMS norm with the weight <c>vision.merger.pool_norm.weight</c> when present (the soft-embedding
/// norm) and <c>vision.merger.out</c>; a fixed square <c>image_size</c> and <c>resize.filter</c> (<c>bicubic</c>, the default | <c>bilinear</c> |
/// <c>lanczos</c>: Pillow's antialiased filters);</item>
/// <item>Idefics3 / SmolVLM towers: <c>tile_size</c> (side of the square tiles in pixels; 0 = none, the default) with <c>resize.longest_edge</c>: the
/// image is resized (aspect preserved, the longer side to longest_edge, the shorter rounded up to even), then to the next multiples of tile_size
/// (aspect preserved again, sides rounded up), and cut into a grid of tiles; the overview is that image resized to one tile. Every tile and the
/// overview are encoded on their own (each one attention window over its patches, the position grid resampled to the tile) and the output rows
/// are the tiles' in raster order, then the overview's. An image that already fits one tile is encoded as the overview alone. The merger then
/// may consist of <c>vision.merger.out</c> only (a linear map of the merged rows, pixel shuffle + projection). The prompt layout
/// (<c>tile.layout</c> <c>idefics3</c>): per tile row, per tile <c>&lt;fake_token_around_image&gt;&lt;row_r_col_c&gt;</c> and its rows, a newline
/// after every tile row but the last; then <c>\n\n&lt;fake_token_around_image&gt;&lt;global-img&gt;</c>, the overview's rows and
/// <c>&lt;fake_token_around_image&gt;</c> (the texts come from <c>image_begin_token</c> and <c>tile.global_token</c>);</item>
/// <item>LFM2-VL towers (<c>tile.layout</c> <c>lfm2</c>, Lfm2VlImageProcessor): an image is smart-resized within <c>min_pixels</c> / <c>max_pixels</c>
/// (the model's own bounds, not the encoder's token limits); one whose area, rounded to whole merge units, exceeds max_pixels ·
/// <c>tile.max_pixels_tolerance</c> (default 2) is instead resized to the tile grid of <c>tile.min_tiles</c>..<c>tile.max_tiles</c> tiles (defaults 2, 10)
/// closest to its aspect ratio, cut into tiles of <c>tile_size</c> and, with <c>tile.use_thumbnail</c> (default true), followed by the smart-resized
/// image as thumbnail. Every view (tile or thumbnail, of its own size) is one attention window with the position grid resampled to it
/// (<c>position_embedding.interpolation</c> <c>bilinear_antialias</c>); after the merge an optional LayerNorm <c>vision.merger.merged_norm</c>
/// (epsilon <c>merger.norm_eps</c>) precedes the MLP. Prompt: <c>image_begin_token</c>, per tile <c>&lt;|img_row_r_col_c|&gt;</c> and its rows,
/// <c>tile.global_token</c> and the thumbnail's rows, <c>image_end_token</c>; an untiled image is only its framed rows;</item>
/// <item>preprocessing: <c>image_mean</c>, <c>image_std</c> (float[in_channels]), <c>min_pixels</c>, <c>max_pixels</c>
/// (long; 0 = unlimited; bounds of the resized image area).</item>
/// </list>
/// <para><b>Weights</b> (<c>.weight</c>/<c>.bias</c> pairs unless noted; linear weights are [out, in]):
/// <c>vision.patch_embed.{weight,bias}</c> — weight [hidden, in_channels·temporal_patch_size·patch_size²], the patch
/// convolution flattened in (channel, frame, row, col) order; <c>vision.position_embedding.weight</c> [count, hidden]; with <c>class_token</c>
/// <c>vision.class_embedding</c> and <c>vision.class_position_embedding</c> [hidden] (their sum is the class row);
/// per layer <c>vision.layers.{i}.</c>: <c>input_norm</c> and <c>post_attention_norm</c> (LayerNorm weight+bias),
/// <c>attention.qkv</c> (fused [3·hidden, hidden], rows ordered q | k | v, each head-major) or separate <c>attention.q</c>, <c>attention.k</c>,
/// <c>attention.v</c>, then <c>attention.o</c>, <c>mlp.up</c>, <c>mlp.down</c> and optionally <c>mlp.gate</c> (gated MLP: act(gate)·up); norms
/// without <c>.bias</c> when <c>norm.type</c> is <c>rms</c>; the patch embedding's bias is optional; optionally <c>vision.patch_norm</c> (a norm of the patch embeddings, before the position embedding is added) and <c>vision.post_norm</c> (a norm of
/// the trunk output, before the merger); the patch merger: optionally <c>vision.merger.norm</c> (applied per patch before merging, epsilon
/// <c>merger.norm_eps</c>), optionally <c>vision.merger.downsample</c> [out_hidden, m²·hidden] (a linear map of the merged row, with bias) and
/// <c>vision.merger.proj</c> + <c>vision.merger.proj_norm</c> (linear, LayerNorm with weight and bias, exact GELU), then the MLP
/// <c>vision.merger.up</c>, optional <c>vision.merger.gate</c> (gated: act(gate)·up) and <c>vision.merger.down</c>, then optionally
/// <c>vision.merger.row_separator</c> [width of down's output] (with <c>merger.row_separator</c>), <c>vision.merger.out</c> (a linear map to
/// out_hidden, with bias, applied to the merged rows and the separator), <c>vision.merger.begin</c> and <c>vision.merger.end</c> [out_hidden]
/// (with <c>merger.boundary</c>) and <c>vision.merger.out_norm</c> (a norm of every output row, merger norm type and epsilon); the Qwen merger has
/// <c>vision.merger.up</c> [m²·hidden, m²·hidden] and <c>vision.merger.down</c> [out_hidden, m²·hidden]; per DeepStack layer i
/// <c>vision.deepstack.{i}.norm</c> (LayerNorm over m²·hidden, applied after merging), <c>vision.deepstack.{i}.up</c> and
/// <c>vision.deepstack.{i}.down</c> (shapes of the merger's; same activation).</para>
/// <para><b>Graph contract</b> (<see cref="WellKnown.VisionEncodeGraph"/>): inputs <see cref="PatchesInput"/> [N, K],
/// <see cref="PositionsInput"/> i32 [N, 2] (row, col), <see cref="EmbeddingIndexInput"/> i32 [N, taps],
/// <see cref="EmbeddingWeightInput"/> f32 [N, taps] (<see cref="PositionTaps"/>), and with windowed attention <see cref="WindowSpansInput"/> i32 [N, 2] (first patch and patch count
/// of each patch's window) and <see cref="WindowOrderInput"/> i32 [M] (for each output row in merge-window raster order, its row in window order); symbols <c>N</c> = patches, <c>M</c> = N / m² (the merger reshapes N → M); output <see cref="EmbeddingsOutput"/> f32 [M, <see cref="EmbeddingWidth"/>]. Patches are ordered in merge-window
/// order, or window by window with windowed attention (see <see cref="VisionPreprocessor"/>); the output rows are always in merge-window raster order.
/// With a row separator or boundary rows the graph also takes <see cref="LayoutInput"/> i32 [R] (symbol <see cref="OutputSymbol"/>): for each output
/// row, its row in the table [merged rows (M) | class row | separator | begin | end] (the entries the model has, in that order); the output is then
/// [R, width].
/// Inputs are produced by <see cref="VisionPreprocessor"/>.</para>
/// </remarks>
public sealed record VisionParameters
{
    internal const string PropertyPrefix = "vision.";
    /// <summary>Name of the patches input of the vision-encode graph, f32 [N, <see cref="PatchDim"/>].</summary>
    public const string PatchesInput = "patches";
    /// <summary>Name of the (row, col) positions input of the vision-encode graph, i32 [N, 2].</summary>
    public const string PositionsInput = "positions";
    /// <summary>Name of the position-embedding tap indices input, i32 [N, 4].</summary>
    public const string EmbeddingIndexInput = "position_embedding_index";
    /// <summary>Name of the position-embedding tap weights input, f32 [N, 4].</summary>
    public const string EmbeddingWeightInput = "position_embedding_weight";
    /// <summary>Name of the windowed-attention spans input, i32 [N, 2]: first patch and patch count of the window of each patch.</summary>
    public const string WindowSpansInput = "window_spans";
    /// <summary>Name of the window-order input, i32 [M]: the window-ordered row of each output row (merge-window raster order).</summary>
    public const string WindowOrderInput = "window_order";
    /// <summary>Name of the output-layout input, i32 [R]: per output row, its row in [merged rows | separator | begin | end].</summary>
    public const string LayoutInput = "output_layout";
    /// <summary>Name of the output of the vision-encode graph, f32 [M, <see cref="EmbeddingWidth"/>].</summary>
    public const string EmbeddingsOutput = "embeddings";
    /// <summary>The image placeholder of the Qwen-VL chat templates, the default <see cref="ImageToken"/>.</summary>
    public const string DefaultImageToken = "<|image_pad|>";
    /// <summary>Symbol of the patch count N.</summary>
    public const string PatchSymbol = "N";
    /// <summary>Symbol of the merged token count M (N divided by <see cref="MergeUnit"/>).</summary>
    public const string MergedSymbol = "M";
    /// <summary>Symbol of the output row count R of a tower with a <see cref="LayoutInput"/>.</summary>
    public const string OutputSymbol = "R";
    /// <summary>Rotation pairs (i, i + head_dim/2), the default <see cref="RopeLayout"/>.</summary>
    public const string PairedRopeLayout = "paired";
    /// <summary>One half of the head per axis, each rotated on its own (Gemma 4), see <see cref="RopeLayout"/>.</summary>
    public const string SplitRopeLayout = "split";
    /// <summary>The aspect-preserving resize within <c>min_pixels</c> / <c>max_pixels</c>, the default <see cref="ResizeMode"/>.</summary>
    public const string SmartResize = "smart";
    /// <summary>The largest image within <c>max_pixels</c> (Gemma 4), see <see cref="ResizeMode"/>.</summary>
    public const string FitBudgetResize = "fit_budget";
    /// <summary>The Qwen-VL M-RoPE image positions, the default <see cref="ImagePositions"/>.</summary>
    public const string MropeImagePositions = "mrope";
    /// <summary>HunyuanVL's sequential image positions (see <see cref="ImagePositions"/>).</summary>
    public const string SequentialImagePositions = "sequential";
    /// <summary>CogVLM's grouped image positions (see <see cref="ImagePositions"/>).</summary>
    public const string GroupedImagePositions = "grouped";
    /// <summary>The √hidden-scaled mean of a pooled merge unit (Gemma 4), the default <see cref="PoolScale"/>.</summary>
    public const string SqrtHiddenPoolScale = "sqrt_hidden";
    /// <summary>The plain mean of a pooled merge unit (Gemma 3), see <see cref="PoolScale"/>.</summary>
    public const string MeanPoolScale = "mean";
    /// <summary>The Idefics3 / SmolVLM prompt layout of a tiled image, see <see cref="TileLayout"/>.</summary>
    public const string Idefics3TileLayout = "idefics3";
    /// <summary>The LFM2-VL tiling and prompt layout, see <see cref="TileLayout"/>.</summary>
    public const string Lfm2TileLayout = "lfm2";
    /// <summary>
    /// PyTorch's <c>interpolate(mode="bilinear", align_corners=False, antialias=True)</c> of the learned grid to the patch grid (SigLIP 2 NaFlex), see
    /// <see cref="PositionInterpolation"/>: every patch has a dense row of weights over the whole grid.
    /// </summary>
    public const string AntialiasedBilinearInterpolation = "bilinear_antialias";
    /// <summary>Every row of an image uses the media weight set, the default <see cref="MediaWeightRows"/>.</summary>
    public const string AllMediaWeightRows = "all";
    /// <summary>Every row of an image but its last uses the media weight set (CogVLM, see <see cref="MediaWeightRows"/>).</summary>
    public const string AllButLastMediaWeightRows = "all_but_last";
    /// <summary>The language model attends causally over an image's rows, the default <see cref="ImageAttention"/>.</summary>
    public const string CausalImageAttention = "causal";
    /// <summary>The rows of one image attend to each other in the language model (Gemma 3), see <see cref="ImageAttention"/>.</summary>
    public const string BidirectionalImageAttention = "bidirectional";

    /// <summary>Number of transformer layers in the tower.</summary>
    public int LayerCount { get; init; }
    /// <summary>Width of the tower's hidden states; a multiple of <see cref="HeadCount"/>.</summary>
    public int HiddenSize { get; init; }
    /// <summary>Width of the feed-forward hidden layer.</summary>
    public int FfnHiddenSize { get; init; }
    /// <summary>Attention heads per layer.</summary>
    public int HeadCount { get; init; }
    /// <summary>Epsilon of the layer norms.</summary>
    public float NormEps { get; init; } = 1e-6f;
    /// <summary>Activation of the trunk MLP: "gelu_tanh", "gelu", "silu" or "quick_gelu".</summary>
    public string Activation { get; init; } = "gelu_tanh";
    /// <summary>Kind of every norm of the tower and merger: "layer" (LayerNorm with bias) or "rms".</summary>
    public string NormType { get; init; } = "layer";
    /// <summary>Where the trunk's norms sit: "pre" (normed sub-layer input) or "post" (normed sub-layer output, EVA2-CLIP).</summary>
    public string NormPlacement { get; init; } = "pre";
    /// <summary>Whether a learned class row joins the patches in the trunk (dropped before the merger output).</summary>
    public bool ClassToken { get; init; }
    /// <summary>Side of the fixed square images are resized to, or 0 for the aspect-preserving resize within the pixel bounds.</summary>
    public int ImageSize { get; init; }
    /// <summary>Side of the square attention windows in merged tokens, or 0 for attention over the whole image in every layer.</summary>
    public int WindowSize { get; init; }
    /// <summary>With <see cref="WindowSize"/>: the layers that attend over the whole image (ascending).</summary>
    public int[] FullAttentionLayers { get; init; } = [];
    /// <summary>Side of a square patch in pixels.</summary>
    public int PatchSize { get; init; }
    /// <summary>Frames per patch; a still image is repeated to fill it.</summary>
    public int TemporalPatchSize { get; init; } = 1;
    /// <summary>Colour channels of the input (3 for RGB).</summary>
    public int InChannels { get; init; } = 3;
    /// <summary>Side m of the window of neighbouring patches merged into one output token.</summary>
    public int SpatialMergeSize { get; init; } = 1;
    /// <summary>Entries of the learned absolute position grid (a perfect square), or 0 for none.</summary>
    public int PositionEmbeddingCount { get; init; }
    /// <summary>
    /// How the learned position grid is resampled to the image grid: "bilinear" (align_corners, 4 taps per patch), "bilinear_half_pixel"
    /// (PyTorch's bilinear resize to the patch grid with align_corners off, 4 taps per patch; HunyuanVL), "bicubic" (PyTorch
    /// grid_sample at the patch centres, align_corners off, border padding, A = −0.75; 16 taps per patch; GLM-4V) or
    /// <see cref="AntialiasedBilinearInterpolation"/> (the antialiased bilinear resize, as dense weights over the grid; LFM2-VL).
    /// </summary>
    public string PositionInterpolation { get; init; } = "bilinear";
    /// <summary>Base of the axial 2-D rotary position embedding, or 0 for none.</summary>
    public float RopeBase { get; init; }
    /// <summary>Rotary layout: <see cref="PairedRopeLayout"/> (rotation pairs (i, i + head_dim/2)) or <see cref="SplitRopeLayout"/> (Gemma 4: one half per axis).</summary>
    public string RopeLayout { get; init; } = PairedRopeLayout;
    /// <summary>Whether the position embedding is two tables (x then y) summed per patch instead of a resampled grid (Gemma 4).</summary>
    public bool AxialPositionEmbedding { get; init; }
    /// <summary>Attention scale; 0 = 1/√head_dim.</summary>
    public float AttentionScale { get; init; }
    /// <summary>Whether every head's query and key are RMS-normed (<c>attention.q_norm</c>, <c>attention.k_norm</c>).</summary>
    public bool QkNorm { get; init; }
    /// <summary>Whether the values are RMS-normed without a weight.</summary>
    public bool ValueNorm { get; init; }
    /// <summary>Whether a merge unit is averaged (and scaled by √hidden) instead of concatenated.</summary>
    public bool MergerPool { get; init; }
    /// <summary>Scale of a pooled merge unit: <see cref="SqrtHiddenPoolScale"/> (Gemma 4) or <see cref="MeanPoolScale"/> (Gemma 3).</summary>
    public string PoolScale { get; init; } = SqrtHiddenPoolScale;
    /// <summary>Filter of the image resize (antialiased when shrinking).</summary>
    public ImageResampleFilter ResizeFilter { get; init; } = ImageResampleFilter.Bicubic;
    /// <summary>Side of the square tiles a tiled tower (Idefics3) cuts the image into, in pixels; 0 = no tiling.</summary>
    public int TileSize { get; init; }
    /// <summary>With <see cref="TileSize"/>: the longer side of the image before it is fitted to whole tiles.</summary>
    public int LongestEdge { get; init; }
    /// <summary>Prompt layout of a tiled image: <see cref="Idefics3TileLayout"/>, or empty without tiling.</summary>
    public string TileLayout { get; init; } = "";
    /// <summary>With <see cref="TileLayout"/>: text of the token that precedes the overview's rows.</summary>
    public string TileGlobalToken { get; init; } = "";
    /// <summary>LFM2 tiling: fewest tiles of a tile grid.</summary>
    public int MinTiles { get; init; } = 2;
    /// <summary>LFM2 tiling: most tiles of a tile grid.</summary>
    public int MaxTiles { get; init; } = 10;
    /// <summary>LFM2 tiling: an image is tiled when its area (rounded to whole merge units) exceeds max_pixels times this factor.</summary>
    public float MaxPixelsTolerance { get; init; } = 2f;
    /// <summary>LFM2 tiling: whether a tiled image is followed by its thumbnail (the smart-resized image).</summary>
    public bool UseThumbnail { get; init; } = true;
    /// <summary>Whether the pooled rows are standardized with <c>vision.std_bias</c> / <c>vision.std_scale</c>.</summary>
    public bool Standardize { get; init; }
    /// <summary>How images are resized: <see cref="SmartResize"/> or <see cref="FitBudgetResize"/>.</summary>
    public string ResizeMode { get; init; } = SmartResize;
    /// <summary>Text of the token that precedes an image's rows in the prompt; empty for none.</summary>
    public string ImageBeginToken { get; init; } = "";
    /// <summary>Text of the token that follows an image's rows in the prompt; empty for none.</summary>
    public string ImageEndToken { get; init; } = "";
    /// <summary>Width of the produced embeddings; equals the hidden size of the language model.</summary>
    public int OutHiddenSize { get; init; }
    /// <summary>Tower layers whose outputs feed DeepStack mergers (ascending); empty for none.</summary>
    public int[] DeepstackLayers { get; init; } = [];
    /// <summary>Activation of the patch merger MLP: "gelu", "gelu_tanh" or "silu".</summary>
    public string MergerActivation { get; init; } = "gelu";
    /// <summary>Epsilon of the patch merger's norms; null = <see cref="NormEps"/>.</summary>
    public float? MergerNormEps { get; init; }
    /// <summary>Kind of the patch merger's norms ("layer" or "rms"); null = <see cref="NormType"/>.</summary>
    public string? MergerNormType { get; init; }
    /// <summary>Whether a learned row follows every row of merged tokens in the output (HunyuanVL).</summary>
    public bool RowSeparator { get; init; }
    /// <summary>Whether learned begin and end rows frame the image's output (HunyuanVL).</summary>
    public bool ImageBoundary { get; init; }
    /// <summary>
    /// How the language model positions an image's rows: <see cref="MropeImagePositions"/>, <see cref="SequentialImagePositions"/> or
    /// <see cref="GroupedImagePositions"/>.
    /// </summary>
    public string ImagePositions { get; init; } = MropeImagePositions;
    /// <summary>Which image rows use the language model's media weight set: <see cref="AllMediaWeightRows"/> or <see cref="AllButLastMediaWeightRows"/>.</summary>
    public string MediaWeightRows { get; init; } = AllMediaWeightRows;
    /// <summary>
    /// How the language model's attention treats an image's rows (key <c>image_attention</c>): <see cref="CausalImageAttention"/>, or
    /// <see cref="BidirectionalImageAttention"/>, where every row of an image also sees the image's later rows (each still sees all earlier
    /// context; transformers' Gemma 3 <c>token_type_ids</c> mask, llama.cpp's non-causal image batches). The multimodal graph then takes the
    /// <see cref="Models.InputRole.AttentionLookahead"/> input.
    /// </summary>
    public string ImageAttention { get; init; } = CausalImageAttention;
    /// <summary>Text of the prompt placeholder that expands to one row per merged image token.</summary>
    public string ImageToken { get; init; } = DefaultImageToken;
    /// <summary>Per-channel mean of the input normalization.</summary>
    public float[] ImageMean { get; init; } = [0.5f, 0.5f, 0.5f];
    /// <summary>Per-channel standard deviation of the input normalization.</summary>
    public float[] ImageStd { get; init; } = [0.5f, 0.5f, 0.5f];
    /// <summary>Lower bound of the resized image area in pixels; 0 = unlimited.</summary>
    public long MinPixels { get; init; }
    /// <summary>Upper bound of the resized image area in pixels; 0 = unlimited.</summary>
    public long MaxPixels { get; init; }
    /// <summary>
    /// A convolutional first stage whose feature map replaces the patch embedding (keys <c>sam.*</c>, DeepSeek-OCR); null for a plain ViT.
    /// With it the image is encoded as fixed-size views by <see cref="SamParameters.PixelsInput"/> graphs instead of patches.
    /// </summary>
    public SamParameters? Sam { get; init; }

    /// <summary>Width of one row of the encoder output: the embedding followed by one DeepStack block per DeepStack layer.</summary>
    public int EmbeddingWidth => OutHiddenSize * (1 + DeepstackLayers.Length);
    /// <summary>Width of one attention head.</summary>
    public int HeadDim => HiddenSize / HeadCount;
    /// <summary>Values per patch: in_channels · temporal_patch_size · patch_size².</summary>
    public int PatchDim => InChannels * TemporalPatchSize * PatchSize * PatchSize;
    /// <summary>Whether the output rows are composed through <see cref="LayoutInput"/> (row separators or boundary rows).</summary>
    public bool HasLayout => RowSeparator || ImageBoundary;

    /// <summary>Rows an image of <paramref name="mergedHeight"/> × <paramref name="mergedWidth"/> merged tokens occupies in the language model.</summary>
    /// <param name="mergedHeight">Rows of merged tokens.</param>
    /// <param name="mergedWidth">Columns of merged tokens.</param>
    public int OutputTokenCount(int mergedHeight, int mergedWidth)
        => mergedHeight * (mergedWidth + (RowSeparator ? 1 : 0)) + (ImageBoundary ? 2 : 0);

    /// <summary>
    /// Taps per patch of the position-embedding resampling (<see cref="PositionInterpolation"/>): 4 or 16; 2 for axial tables; the grid size for
    /// <see cref="AntialiasedBilinearInterpolation"/>.
    /// </summary>
    public int PositionTaps => AxialPositionEmbedding ? 2 : PositionInterpolation == "bicubic" ? 16
        : PositionInterpolation == AntialiasedBilinearInterpolation ? PositionEmbeddingCount : 4;
    /// <summary>Whether the position embedding is the dense product <c>EmbeddingWeight</c> · table (no tap indices, see <see cref="AntialiasedBilinearInterpolation"/>).</summary>
    public bool DensePositionWeights => !AxialPositionEmbedding && PositionInterpolation == AntialiasedBilinearInterpolation;
    /// <summary>Side of the learned position grid.</summary>
    public int PositionGridSide => (int)Math.Round(Math.Sqrt(PositionEmbeddingCount));
    /// <summary>Tiled towers: output rows of the largest tile grid (the longest edge in whole tiles on both sides) plus the overview.</summary>
    public int MaxTiledTokens
    {
        get
        {
            if (TileSize <= 0)
            {
                return 0;
            }

            if (TileLayout == Lfm2TileLayout)
            {
                // The largest grid's tiles plus the thumbnail (at most max_pixels).
                int tileTokens = TileSize / (PatchSize * SpatialMergeSize);
                long unit = (long)PatchSize * SpatialMergeSize * PatchSize * SpatialMergeSize;
                return MaxTiles * tileTokens * tileTokens + (int)(MaxPixels / unit);
            }

            int side = (Math.Min(LongestEdge, 4096) + TileSize - 1) / TileSize;
            int perView = TileSize / (PatchSize * SpatialMergeSize);
            return (side * side + 1) * perView * perView;
        }
    }

    /// <summary>Patches merged into one output token (m²).</summary>
    public int MergeUnit => SpatialMergeSize * SpatialMergeSize;

    /// <summary>Reads <c>{prefix}{key}</c> entries; null when <c>layer_count</c> is absent (no vision tower).</summary>
    public static VisionParameters? FromMetadata(IReadOnlyDictionary<string, object> md, string prefix)
    {
        if (!md.ContainsKey(prefix + "layer_count"))
        {
            return null;
        }

        T Get<T>(string k, T d) => md.TryGetValue(prefix + k, out var v) ? Convert<T>(v) : d;
        var p = new VisionParameters
        {
            LayerCount = Get<int>("layer_count", 0),
            HiddenSize = Get<int>("hidden_size", 0),
            FfnHiddenSize = Get<int>("ffn_hidden_size", 0),
            HeadCount = Get<int>("head_count", 1),
            NormEps = Get("norm_eps", 1e-6f),
            Activation = Get("activation", "gelu_tanh"),
            NormType = Get("norm.type", "layer"),
            NormPlacement = Get("norm.placement", "pre"),
            ClassToken = Get("class_token", false),
            ImageSize = Get("image_size", 0),
            WindowSize = Get("attention.window", 0),
            FullAttentionLayers = Get("attention.full_layers", Array.Empty<int>()),
            PatchSize = Get<int>("patch_size", 0),
            TemporalPatchSize = Get("temporal_patch_size", 1),
            InChannels = Get("in_channels", 3),
            SpatialMergeSize = Get("spatial_merge_size", 1),
            PositionEmbeddingCount = Get("position_embedding_count", 0),
            PositionInterpolation = Get("position_embedding.interpolation", "bilinear"),
            RopeBase = Get("rope.base", 0f),
            RopeLayout = Get("rope.layout", PairedRopeLayout),
            AxialPositionEmbedding = Get("position_embedding.axial", false),
            AttentionScale = Get("attention.scale", 0f),
            QkNorm = Get("attention.qk_norm", false),
            ValueNorm = Get("attention.v_norm", false),
            MergerPool = Get("merger.pool", false),
            Standardize = Get("standardize", false),
            PoolScale = Get("merger.pool.scale", SqrtHiddenPoolScale),
            ResizeFilter = Enum.Parse<ImageResampleFilter>(Get("resize.filter", "bicubic"), ignoreCase: true),
            TileSize = Get("tile_size", 0),
            LongestEdge = Get("resize.longest_edge", 0),
            TileLayout = Get("tile.layout", ""),
            TileGlobalToken = Get("tile.global_token", ""),
            MinTiles = Get("tile.min_tiles", 2),
            MaxTiles = Get("tile.max_tiles", 10),
            MaxPixelsTolerance = Get("tile.max_pixels_tolerance", 2f),
            UseThumbnail = Get("tile.use_thumbnail", true),
            ResizeMode = Get("resize.mode", SmartResize),
            ImageBeginToken = Get("image_begin_token", ""),
            ImageEndToken = Get("image_end_token", ""),
            OutHiddenSize = Get<int>("out_hidden_size", 0),
            MergerActivation = Get("merger.activation", "gelu"),
            MergerNormEps = md.ContainsKey(prefix + "merger.norm_eps") ? Get("merger.norm_eps", 0f) : null,
            MergerNormType = md.ContainsKey(prefix + "merger.norm.type") ? Get("merger.norm.type", "layer") : null,
            RowSeparator = Get("merger.row_separator", false),
            ImageBoundary = Get("merger.boundary", false),
            ImagePositions = Get("image_positions", MropeImagePositions),
            MediaWeightRows = Get("media_weight_rows", AllMediaWeightRows),
            ImageAttention = Get("image_attention", CausalImageAttention),
            ImageToken = Get("image_token", DefaultImageToken),
            DeepstackLayers = Get("deepstack_layers", Array.Empty<int>()),
            ImageMean = Get("image_mean", new[] { 0.5f, 0.5f, 0.5f }),
            ImageStd = Get("image_std", new[] { 0.5f, 0.5f, 0.5f }),
            MinPixels = Get("min_pixels", 0L),
            MaxPixels = Get("max_pixels", 0L),
            Sam = SamParameters.FromMetadata(md, prefix),
        };
        p.Validate();
        return p;
    }

    /// <summary>Reads the tower of a built model (<c>vision.*</c> properties); null if the model has none.</summary>
    public static VisionParameters? FromProperties(IReadOnlyDictionary<string, object> props) => FromMetadata(props, PropertyPrefix);

    /// <summary>The tower of a built model, or null.</summary>
    /// <param name="model">The built model.</param>
    public static VisionParameters? Of(ModelDefinition model) => FromProperties(model.Properties);

    /// <summary>Adds the parameters as <c>vision.*</c> properties (what packages store).</summary>
    /// <param name="props">Receives the <c>vision.*</c> entries.</param>
    public void WriteProperties(IDictionary<string, object> props)
    {
        void Set(string k, object v) => props[PropertyPrefix + k] = v;
        Set("layer_count", LayerCount);
        Set("hidden_size", HiddenSize);
        Set("ffn_hidden_size", FfnHiddenSize);
        Set("head_count", HeadCount);
        Set("norm_eps", NormEps);
        Set("activation", Activation);
        Set("norm.type", NormType);
        if (NormPlacement != "pre")
        {
            Set("norm.placement", NormPlacement);
        }

        if (ClassToken)
        {
            Set("class_token", true);
        }

        if (ImageSize > 0)
        {
            Set("image_size", ImageSize);
        }

        if (WindowSize > 0)
        {
            Set("attention.window", WindowSize);
            Set("attention.full_layers", FullAttentionLayers);
        }

        Set("patch_size", PatchSize);
        Set("temporal_patch_size", TemporalPatchSize);
        Set("in_channels", InChannels);
        Set("spatial_merge_size", SpatialMergeSize);
        Set("position_embedding_count", PositionEmbeddingCount);
        if (PositionInterpolation != "bilinear")
        {
            Set("position_embedding.interpolation", PositionInterpolation);
        }
        Set("rope.base", RopeBase);
        if (RopeLayout != PairedRopeLayout)
        {
            Set("rope.layout", RopeLayout);
        }

        if (AxialPositionEmbedding)
        {
            Set("position_embedding.axial", true);
        }

        if (AttentionScale != 0f)
        {
            Set("attention.scale", AttentionScale);
        }

        if (QkNorm)
        {
            Set("attention.qk_norm", true);
        }

        if (ValueNorm)
        {
            Set("attention.v_norm", true);
        }

        if (MergerPool)
        {
            Set("merger.pool", true);
        }

        if (Standardize)
        {
            Set("standardize", true);
        }

        if (ResizeMode != SmartResize)
        {
            Set("resize.mode", ResizeMode);
        }

        if (PoolScale != SqrtHiddenPoolScale)
        {
            Set("merger.pool.scale", PoolScale);
        }

        if (ResizeFilter != ImageResampleFilter.Bicubic)
        {
            Set("resize.filter", ResizeFilter.ToString().ToLowerInvariant());
        }

        if (TileSize > 0)
        {
            Set("tile_size", TileSize);
            Set("resize.longest_edge", LongestEdge);
            Set("tile.layout", TileLayout);
            Set("tile.global_token", TileGlobalToken);
            if (TileLayout == Lfm2TileLayout)
            {
                Set("tile.min_tiles", MinTiles);
                Set("tile.max_tiles", MaxTiles);
                Set("tile.max_pixels_tolerance", MaxPixelsTolerance);
                Set("tile.use_thumbnail", UseThumbnail);
            }
        }

        if (ImageBeginToken.Length > 0)
        {
            Set("image_begin_token", ImageBeginToken);
        }

        if (ImageEndToken.Length > 0)
        {
            Set("image_end_token", ImageEndToken);
        }

        Set("out_hidden_size", OutHiddenSize);
        Set("merger.activation", MergerActivation);
        if (MergerNormEps is float mergerEps)
        {
            Set("merger.norm_eps", mergerEps);
        }

        if (MergerNormType is { } mergerNormType)
        {
            Set("merger.norm.type", mergerNormType);
        }

        if (RowSeparator)
        {
            Set("merger.row_separator", true);
        }

        if (ImageBoundary)
        {
            Set("merger.boundary", true);
        }

        if (ImagePositions != MropeImagePositions)
        {
            Set("image_positions", ImagePositions);
        }

        if (MediaWeightRows != AllMediaWeightRows)
        {
            Set("media_weight_rows", MediaWeightRows);
        }

        if (ImageAttention != CausalImageAttention)
        {
            Set("image_attention", ImageAttention);
        }

        if (ImageToken != DefaultImageToken)
        {
            Set("image_token", ImageToken);
        }
        if (DeepstackLayers.Length > 0)
        {
            Set("deepstack_layers", DeepstackLayers);
        }

        Set("image_mean", ImageMean);
        Set("image_std", ImageStd);
        Set("min_pixels", MinPixels);
        Set("max_pixels", MaxPixels);
        Sam?.Write(Set);
    }

    private void Validate()
    {
        if (HiddenSize <= 0 || HeadCount <= 0 || HiddenSize % HeadCount != 0)
        {
            throw new InvalidDataException($"vision: hidden_size {HiddenSize} must be a multiple of head_count {HeadCount}.");
        }

        if (PatchSize <= 0 || SpatialMergeSize <= 0 || OutHiddenSize <= 0)
        {
            throw new InvalidDataException("vision: patch_size, spatial_merge_size and out_hidden_size must be positive.");
        }

        if (RopeBase > 0 && HeadDim % 4 != 0)
        {
            throw new InvalidDataException($"vision: 2-D rotary embedding needs head_dim % 4 == 0 (head_dim {HeadDim}).");
        }

        if (ImageMean.Length != InChannels || ImageStd.Length != InChannels)
        {
            throw new InvalidDataException("vision: image_mean/std must have in_channels entries.");
        }

        for (int i = 0; i < DeepstackLayers.Length; i++)
        {
            if (DeepstackLayers[i] < 0 || DeepstackLayers[i] >= LayerCount || (i > 0 && DeepstackLayers[i] <= DeepstackLayers[i - 1]))
            {
                throw new InvalidDataException("vision: deepstack_layers must be ascending tower layer indices.");
            }
        }

        if (NormType is not ("layer" or "rms") || MergerNormType is not (null or "layer" or "rms"))
        {
            throw new InvalidDataException($"vision: norm.type '{NormType}' / merger.norm.type '{MergerNormType}' is not supported (layer, rms).");
        }

        if (ImagePositions is not (MropeImagePositions or SequentialImagePositions or GroupedImagePositions))
        {
            throw new InvalidDataException(
                $"vision: image_positions '{ImagePositions}' is not supported ({MropeImagePositions}, {SequentialImagePositions}, {GroupedImagePositions}).");
        }

        if (MediaWeightRows is not (AllMediaWeightRows or AllButLastMediaWeightRows))
        {
            throw new InvalidDataException($"vision: media_weight_rows '{MediaWeightRows}' is not supported ({AllMediaWeightRows}, {AllButLastMediaWeightRows}).");
        }

        if (ImageAttention is not (CausalImageAttention or BidirectionalImageAttention))
        {
            throw new InvalidDataException($"vision: image_attention '{ImageAttention}' is not supported ({CausalImageAttention}, {BidirectionalImageAttention}).");
        }

        if (NormPlacement is not ("pre" or "post" or "sandwich") || ImageSize < 0)
        {
            throw new InvalidDataException($"vision: norm.placement '{NormPlacement}' is not supported (pre, post, sandwich) or image_size {ImageSize} is negative.");
        }

        if (RopeLayout is not (PairedRopeLayout or SplitRopeLayout) || ResizeMode is not (SmartResize or FitBudgetResize))
        {
            throw new InvalidDataException($"vision: rope.layout '{RopeLayout}' (paired, split) or resize.mode '{ResizeMode}' (smart, fit_budget) is not supported.");
        }

        if (ClassToken && (!HasLayout || WindowSize > 0 || RopeBase > 0))
        {
            throw new InvalidDataException("vision: a class token needs an output layout (boundary rows or row separators), no windowed attention and no rotary embedding.");
        }

        if (HasLayout && DeepstackLayers.Length > 0)
        {
            throw new InvalidDataException("vision: row separators and boundary rows are not supported together with DeepStack layers.");
        }

        if (WindowSize < 0 || FullAttentionLayers.Any(l => l < 0 || l >= LayerCount))
        {
            throw new InvalidDataException("vision: attention.window must be >= 0 and attention.full_layers tower layer indices.");
        }

        if (PositionInterpolation is not ("bilinear" or "bilinear_half_pixel" or "bicubic" or AntialiasedBilinearInterpolation))
        {
            throw new InvalidDataException(
                $"vision: position_embedding.interpolation '{PositionInterpolation}' is not supported (bilinear, bilinear_half_pixel, bicubic, {AntialiasedBilinearInterpolation}).");
        }

        if (PoolScale is not (SqrtHiddenPoolScale or MeanPoolScale) || ResizeFilter is not (ImageResampleFilter.Bilinear or ImageResampleFilter.Bicubic or ImageResampleFilter.Lanczos))
        {
            throw new InvalidDataException($"vision: merger.pool.scale '{PoolScale}' ({SqrtHiddenPoolScale}, {MeanPoolScale}) or resize.filter {ResizeFilter} (bilinear, bicubic, lanczos) is not supported.");
        }

        if (TileSize < 0 || (TileSize > 0 && (TileLayout is not (Idefics3TileLayout or Lfm2TileLayout) || TileSize % (PatchSize * SpatialMergeSize) != 0
            || (TileLayout == Idefics3TileLayout && LongestEdge < TileSize) || (TileLayout == Lfm2TileLayout && (MinTiles < 1 || MaxTiles < MinTiles || MaxPixels <= 0))
            || WindowSize > 0 || HasLayout || ClassToken || DeepstackLayers.Length > 0 || Sam is not null || RopeBase > 0)))
        {
            throw new InvalidDataException(
                $"vision: tile_size {TileSize} needs tile.layout '{Idefics3TileLayout}' (with resize.longest_edge >= tile_size) or '{Lfm2TileLayout}' (with 1 <= tile.min_tiles <= tile.max_tiles and max_pixels), a multiple of patch_size·spatial_merge_size and a plain tower (no windows, layout rows, class token, DeepStack, SAM stage or rotary embedding).");
        }

        if (PositionEmbeddingCount > 0 && !AxialPositionEmbedding && PositionGridSide * PositionGridSide != PositionEmbeddingCount)
        {
            throw new InvalidDataException("vision: position_embedding_count must be a square.");
        }
    }

    internal static T Convert<T>(object v)
    {
        if (v is T t)
        {
            return t;
        }

        if (typeof(T).IsArray && v is Array src)
        {
            var et = typeof(T).GetElementType()!;
            var dst = Array.CreateInstance(et, src.Length);
            for (int i = 0; i < src.Length; i++)
            {
                dst.SetValue(System.Convert.ChangeType(src.GetValue(i), et, CultureInfo.InvariantCulture), i);
            }

            return (T)(object)dst;
        }
        return (T)System.Convert.ChangeType(v, typeof(T), CultureInfo.InvariantCulture);
    }
}
