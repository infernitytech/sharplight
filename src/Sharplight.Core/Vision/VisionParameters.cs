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
/// <c>norm.type</c> (<c>layer</c>, the default: LayerNorm with weight and bias | <c>rms</c>: RMS norm with a weight; every norm of the tower and merger);</item>
/// <item><c>attention.window</c> (side of the square attention windows in merged tokens; 0 = none, the default) with <c>attention.full_layers</c>
/// (int[], ascending layers that attend over the whole image): windowed attention (Qwen2.5-VL). The merged-token grid is cut into windows of
/// window×window merged tokens (the last row/column of windows is smaller when the grid is not a multiple), patches are ordered window by
/// window and a patch of a windowed layer only attends to the patches of its window;</item>
/// <item><c>patch_size</c>, <c>temporal_patch_size</c> (frames per patch; a still image is repeated), <c>in_channels</c>,
/// <c>spatial_merge_size</c> (m: m×m neighbouring patches are merged into one output token);</item>
/// <item><c>position_embedding_count</c> (learned absolute position grid of g×g entries, g² = count, bilinearly
/// resampled to the image grid with align_corners; 0 = none), <c>rope.base</c> (axial 2-D rotary position
/// embedding over (row, col), 0 = none);</item>
/// <item><c>out_hidden_size</c> (width of the produced embeddings = hidden size of the language model),
/// <c>merger.activation</c> (<c>gelu</c> | <c>gelu_tanh</c> | <c>silu</c>);</item>
/// <item><c>deepstack_layers</c> (int[], ascending tower layer indices; default none): DeepStack (Qwen3-VL). The output of each listed
/// layer is merged like the final hidden states and projected by its own merger; the results are appended to the embedding row, which is
/// then <see cref="EmbeddingWidth"/> = out_hidden_size·(1 + count) wide. The language model adds the k-th appended block to the hidden
/// state after its k-th layer;</item>
/// <item>preprocessing: <c>image_mean</c>, <c>image_std</c> (float[in_channels]), <c>min_pixels</c>, <c>max_pixels</c>
/// (long; 0 = unlimited; bounds of the resized image area).</item>
/// </list>
/// <para><b>Weights</b> (<c>.weight</c>/<c>.bias</c> pairs unless noted; linear weights are [out, in]):
/// <c>vision.patch_embed.{weight,bias}</c> — weight [hidden, in_channels·temporal_patch_size·patch_size²], the patch
/// convolution flattened in (channel, frame, row, col) order; <c>vision.position_embedding.weight</c> [count, hidden];
/// per layer <c>vision.layers.{i}.</c>: <c>input_norm</c> and <c>post_attention_norm</c> (LayerNorm weight+bias),
/// <c>attention.qkv</c> (fused [3·hidden, hidden], rows ordered q | k | v, each head-major) or separate <c>attention.q</c>, <c>attention.k</c>,
/// <c>attention.v</c>, then <c>attention.o</c>, <c>mlp.up</c>, <c>mlp.down</c> and optionally <c>mlp.gate</c> (gated MLP: act(gate)·up); norms
/// without <c>.bias</c> when <c>norm.type</c> is <c>rms</c>; the patch embedding's bias is optional; the patch merger <c>vision.merger.norm</c> (LayerNorm over hidden, applied per patch before merging),
/// <c>vision.merger.up</c> [m²·hidden, m²·hidden] and <c>vision.merger.down</c> [out_hidden, m²·hidden]; per DeepStack layer i
/// <c>vision.deepstack.{i}.norm</c> (LayerNorm over m²·hidden, applied after merging), <c>vision.deepstack.{i}.up</c> and
/// <c>vision.deepstack.{i}.down</c> (shapes of the merger's; same activation).</para>
/// <para><b>Graph contract</b> (<see cref="WellKnown.VisionEncodeGraph"/>): inputs <see cref="PatchesInput"/> [N, K],
/// <see cref="PositionsInput"/> i32 [N, 2] (row, col), <see cref="EmbeddingIndexInput"/> i32 [N, 4],
/// <see cref="EmbeddingWeightInput"/> f32 [N, 4], and with windowed attention <see cref="WindowSpansInput"/> i32 [N, 2] (first patch and patch count
/// of each patch's window) and <see cref="WindowOrderInput"/> i32 [M] (for each output row in merge-window raster order, its row in window order); symbols <c>N</c> = patches, <c>M</c> = N / m² (the merger reshapes N → M); output <see cref="EmbeddingsOutput"/> f32 [M, <see cref="EmbeddingWidth"/>]. Patches are ordered in merge-window
/// order, or window by window with windowed attention (see <see cref="VisionPreprocessor"/>); the output rows are always in merge-window raster order. Inputs are produced by <see cref="VisionPreprocessor"/>.</para>
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
    /// <summary>Name of the output of the vision-encode graph, f32 [M, <see cref="EmbeddingWidth"/>].</summary>
    public const string EmbeddingsOutput = "embeddings";
    /// <summary>Symbol of the patch count N.</summary>
    public const string PatchSymbol = "N";
    /// <summary>Symbol of the merged token count M (N divided by <see cref="MergeUnit"/>).</summary>
    public const string MergedSymbol = "M";

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
    /// <summary>Base of the axial 2-D rotary position embedding, or 0 for none.</summary>
    public float RopeBase { get; init; }
    /// <summary>Width of the produced embeddings; equals the hidden size of the language model.</summary>
    public int OutHiddenSize { get; init; }
    /// <summary>Tower layers whose outputs feed DeepStack mergers (ascending); empty for none.</summary>
    public int[] DeepstackLayers { get; init; } = [];
    /// <summary>Activation of the patch merger MLP: "gelu", "gelu_tanh" or "silu".</summary>
    public string MergerActivation { get; init; } = "gelu";
    /// <summary>Per-channel mean of the input normalization.</summary>
    public float[] ImageMean { get; init; } = [0.5f, 0.5f, 0.5f];
    /// <summary>Per-channel standard deviation of the input normalization.</summary>
    public float[] ImageStd { get; init; } = [0.5f, 0.5f, 0.5f];
    /// <summary>Lower bound of the resized image area in pixels; 0 = unlimited.</summary>
    public long MinPixels { get; init; }
    /// <summary>Upper bound of the resized image area in pixels; 0 = unlimited.</summary>
    public long MaxPixels { get; init; }

    /// <summary>Width of one row of the encoder output: the embedding followed by one DeepStack block per DeepStack layer.</summary>
    public int EmbeddingWidth => OutHiddenSize * (1 + DeepstackLayers.Length);
    /// <summary>Width of one attention head.</summary>
    public int HeadDim => HiddenSize / HeadCount;
    /// <summary>Values per patch: in_channels · temporal_patch_size · patch_size².</summary>
    public int PatchDim => InChannels * TemporalPatchSize * PatchSize * PatchSize;
    /// <summary>Side of the learned position grid.</summary>
    public int PositionGridSide => (int)Math.Round(Math.Sqrt(PositionEmbeddingCount));
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
            WindowSize = Get("attention.window", 0),
            FullAttentionLayers = Get("attention.full_layers", Array.Empty<int>()),
            PatchSize = Get<int>("patch_size", 0),
            TemporalPatchSize = Get("temporal_patch_size", 1),
            InChannels = Get("in_channels", 3),
            SpatialMergeSize = Get("spatial_merge_size", 1),
            PositionEmbeddingCount = Get("position_embedding_count", 0),
            RopeBase = Get("rope.base", 0f),
            OutHiddenSize = Get<int>("out_hidden_size", 0),
            MergerActivation = Get("merger.activation", "gelu"),
            DeepstackLayers = Get("deepstack_layers", Array.Empty<int>()),
            ImageMean = Get("image_mean", new[] { 0.5f, 0.5f, 0.5f }),
            ImageStd = Get("image_std", new[] { 0.5f, 0.5f, 0.5f }),
            MinPixels = Get("min_pixels", 0L),
            MaxPixels = Get("max_pixels", 0L),
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
        Set("rope.base", RopeBase);
        Set("out_hidden_size", OutHiddenSize);
        Set("merger.activation", MergerActivation);
        if (DeepstackLayers.Length > 0)
        {
            Set("deepstack_layers", DeepstackLayers);
        }

        Set("image_mean", ImageMean);
        Set("image_std", ImageStd);
        Set("min_pixels", MinPixels);
        Set("max_pixels", MaxPixels);
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

        if (NormType is not ("layer" or "rms"))
        {
            throw new InvalidDataException($"vision: norm.type '{NormType}' is not supported (layer, rms).");
        }

        if (WindowSize < 0 || FullAttentionLayers.Any(l => l < 0 || l >= LayerCount))
        {
            throw new InvalidDataException("vision: attention.window must be >= 0 and attention.full_layers tower layer indices.");
        }

        if (PositionEmbeddingCount > 0 && PositionGridSide * PositionGridSide != PositionEmbeddingCount)
        {
            throw new InvalidDataException("vision: position_embedding_count must be a square.");
        }
    }

    private static T Convert<T>(object v)
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
