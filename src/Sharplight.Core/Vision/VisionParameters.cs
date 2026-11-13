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
/// <c>activation</c> (<c>gelu_tanh</c> | <c>gelu</c> | <c>silu</c>; trunk MLP);</item>
/// <item><c>patch_size</c>, <c>temporal_patch_size</c> (frames per patch; a still image is repeated), <c>in_channels</c>,
/// <c>spatial_merge_size</c> (m: m×m neighbouring patches are merged into one output token);</item>
/// <item><c>position_embedding_count</c> (learned absolute position grid of g×g entries, g² = count, bilinearly
/// resampled to the image grid with align_corners; 0 = none), <c>rope.base</c> (axial 2-D rotary position
/// embedding over (row, col), 0 = none);</item>
/// <item><c>out_hidden_size</c> (width of the produced embeddings = hidden size of the language model),
/// <c>merger.activation</c> (<c>gelu</c> | <c>gelu_tanh</c> | <c>silu</c>);</item>
/// <item>preprocessing: <c>image_mean</c>, <c>image_std</c> (float[in_channels]), <c>min_pixels</c>, <c>max_pixels</c>
/// (long; 0 = unlimited; bounds of the resized image area).</item>
/// </list>
/// <para><b>Weights</b> (<c>.weight</c>/<c>.bias</c> pairs unless noted; linear weights are [out, in]):
/// <c>vision.patch_embed.{weight,bias}</c> — weight [hidden, in_channels·temporal_patch_size·patch_size²], the patch
/// convolution flattened in (channel, frame, row, col) order; <c>vision.position_embedding.weight</c> [count, hidden];
/// per layer <c>vision.layers.{i}.</c>: <c>input_norm</c> and <c>post_attention_norm</c> (LayerNorm weight+bias),
/// <c>attention.qkv</c> (fused [3·hidden, hidden], rows ordered q | k | v, each head-major), <c>attention.o</c>,
/// <c>mlp.up</c>, <c>mlp.down</c>; the patch merger <c>vision.merger.norm</c> (LayerNorm over hidden, applied per patch before merging),
/// <c>vision.merger.up</c> [m²·hidden, m²·hidden] and <c>vision.merger.down</c> [out_hidden, m²·hidden].</para>
/// <para><b>Graph contract</b> (<see cref="WellKnown.VisionEncodeGraph"/>): inputs <see cref="PatchesInput"/> [N, K],
/// <see cref="PositionsInput"/> i32 [N, 2] (row, col), <see cref="EmbeddingIndexInput"/> i32 [N, 4],
/// <see cref="EmbeddingWeightInput"/> f32 [N, 4]; symbols <c>N</c> = patches, <c>M</c> = N / m² (the merger reshapes N → M); output <see cref="EmbeddingsOutput"/> f32 [M, out_hidden]. Patches are ordered in merge-window
/// order (see <see cref="VisionPreprocessor"/>). Inputs are produced by <see cref="VisionPreprocessor"/>.</para>
/// </remarks>
public sealed record VisionParameters
{
    public const string PropertyPrefix = "vision.";
    public const string PatchesInput = "patches";
    public const string PositionsInput = "positions";
    public const string EmbeddingIndexInput = "position_embedding_index";
    public const string EmbeddingWeightInput = "position_embedding_weight";
    public const string EmbeddingsOutput = "embeddings";
    public const string PatchSymbol = "N";
    public const string MergedSymbol = "M";

    public int LayerCount { get; init; }
    public int HiddenSize { get; init; }
    public int FfnHiddenSize { get; init; }
    public int HeadCount { get; init; }
    public float NormEps { get; init; } = 1e-6f;
    public string Activation { get; init; } = "gelu_tanh";
    public int PatchSize { get; init; }
    public int TemporalPatchSize { get; init; } = 1;
    public int InChannels { get; init; } = 3;
    public int SpatialMergeSize { get; init; } = 1;
    public int PositionEmbeddingCount { get; init; }
    public float RopeBase { get; init; }
    public int OutHiddenSize { get; init; }
    public string MergerActivation { get; init; } = "gelu";
    public float[] ImageMean { get; init; } = [0.5f, 0.5f, 0.5f];
    public float[] ImageStd { get; init; } = [0.5f, 0.5f, 0.5f];
    public long MinPixels { get; init; }
    public long MaxPixels { get; init; }

    public int HeadDim => HiddenSize / HeadCount;
    /// <summary>Values per patch: in_channels · temporal_patch_size · patch_size².</summary>
    public int PatchDim => InChannels * TemporalPatchSize * PatchSize * PatchSize;
    /// <summary>Side of the learned position grid.</summary>
    public int PositionGridSide => (int)Math.Round(Math.Sqrt(PositionEmbeddingCount));
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
            PatchSize = Get<int>("patch_size", 0),
            TemporalPatchSize = Get("temporal_patch_size", 1),
            InChannels = Get("in_channels", 3),
            SpatialMergeSize = Get("spatial_merge_size", 1),
            PositionEmbeddingCount = Get("position_embedding_count", 0),
            RopeBase = Get("rope.base", 0f),
            OutHiddenSize = Get<int>("out_hidden_size", 0),
            MergerActivation = Get("merger.activation", "gelu"),
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
    public static VisionParameters? Of(ModelDefinition model) => FromProperties(model.Properties);

    /// <summary>Adds the parameters as <c>vision.*</c> properties (what packages store).</summary>
    public void WriteProperties(IDictionary<string, object> props)
    {
        void Set(string k, object v) => props[PropertyPrefix + k] = v;
        Set("layer_count", LayerCount); Set("hidden_size", HiddenSize); Set("ffn_hidden_size", FfnHiddenSize);
        Set("head_count", HeadCount); Set("norm_eps", NormEps); Set("activation", Activation);
        Set("patch_size", PatchSize); Set("temporal_patch_size", TemporalPatchSize); Set("in_channels", InChannels);
        Set("spatial_merge_size", SpatialMergeSize); Set("position_embedding_count", PositionEmbeddingCount);
        Set("rope.base", RopeBase); Set("out_hidden_size", OutHiddenSize); Set("merger.activation", MergerActivation);
        Set("image_mean", ImageMean); Set("image_std", ImageStd); Set("min_pixels", MinPixels); Set("max_pixels", MaxPixels);
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
