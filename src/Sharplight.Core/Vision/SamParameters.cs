namespace Sharplight.Vision;

/// <summary>
/// The first stage of a two-stage image tower (<see cref="VisionParameters.Sam"/>, DeepSeek-OCR): a Segment-Anything-style ViT backbone (ViTDet)
/// over <see cref="VisionParameters.PatchSize"/> patches with a learned position grid, windowed attention with a decomposed relative position
/// bias (MViTv2; a few layers attend over the whole view) and a convolution neck, followed by stride-2 convolutions. Its feature map is the
/// patch embedding of the second stage, a ViT described by the rest of <see cref="VisionParameters"/>. The image is cut into views of fixed size,
/// a global view (resized to fit, padded to a square) and, for larger images, local tiles; each view is encoded on its own.
/// </summary>
/// <remarks>
/// Canonical keys below <c>{architecture}.vision.sam.</c>: <c>layer_count</c>, <c>hidden_size</c>, <c>head_count</c>, <c>ffn_hidden_size</c>,
/// <c>norm_eps</c>, <c>window</c>, <c>global_layers</c> (int[]), <c>position_grid</c>, <c>neck_channels</c>, <c>downsample_channels</c> (int[]),
/// <c>global_size</c>, <c>tile_size</c>, <c>min_tiles</c>, <c>max_tiles</c>, <c>pad_value</c>, and for a query encoder <c>encoder</c>
/// (<c>queries</c>; default <c>clip</c>), <c>query_kv_heads</c>, <c>query_rope_base</c>.
/// <para>Weights of the first stage: <c>vision.sam.patch_embed.{weight [hidden, 3, p, p], bias}</c> (a stride-p convolution),
/// <c>vision.sam.position_embedding.weight</c> [grid², hidden] (row-major grid; resampled with antialiased bicubic interpolation for views of
/// another size), per layer <c>vision.sam.layers.{i}.</c> <c>input_norm</c>, <c>post_attention_norm</c> (LayerNorms), <c>attention.qkv</c>
/// ([3·hidden, hidden], q | k | v, each head-major), <c>attention.o</c>, <c>attention.rel_pos_h.weight</c> and <c>attention.rel_pos_w.weight</c>
/// [2·side − 1, head_dim] (side: the window, or the position grid for layers over the whole view; resampled linearly for other sides),
/// <c>mlp.up</c>, <c>mlp.down</c> (exact GELU); the neck <c>vision.sam.neck.conv1.weight</c> [neck, hidden, 1, 1], <c>vision.sam.neck.norm1</c>,
/// <c>vision.sam.neck.conv2.weight</c> [neck, neck, 3, 3] (padding 1), <c>vision.sam.neck.norm2</c> (LayerNorms over the channels), then
/// <c>vision.sam.downsample.{j}.weight</c> [channels_j, in, 3, 3] (stride 2, padding 1). Convolutions have no bias unless noted.</para>
/// <para>Second stage: a ViT whose patch embeddings are the feature map (its width, the last downsample width, is
/// <see cref="VisionParameters.HiddenSize"/>): <c>vision.class_embedding.weight</c> and <c>vision.class_position.weight</c> [1, hidden] (the
/// class token and its position), <c>vision.position_embedding.weight</c> [grid², hidden] (resampled like the first stage's),
/// <c>vision.pre_norm</c>, then the <c>vision.layers.{i}.*</c> of <see cref="VisionParameters"/> (bidirectional). The class row is dropped,
/// each remaining row becomes [second-stage output | feature map] and the projector <c>vision.projector.{weight,bias}</c> maps it to the
/// language model's width.</para>
/// <para>Second stage with <see cref="Encoder"/> <c>queries</c> (DeepSeek-OCR-2): a decoder-style encoder over [feature map | learned
/// queries] (<c>vision.query_embedding.{n}.weight</c> [n, hidden], one set per feature count n): the <c>vision.layers.{i}.*</c> of
/// <see cref="VisionParameters"/> (RMS norms, separate q / k / v projections with <see cref="QueryKvHeads"/> key-value heads, 1-D rotary
/// embedding with base <see cref="QueryRopeBase"/> over positions 0, 1, … rotating the two head halves, gated MLP), where the feature rows attend
/// to each other and every query row to all feature rows and the queries up to itself; then <c>vision.post_norm</c>; the query rows go through
/// the projector. Views have no newline rows: the tiles' rows tile by tile, the global view's, the separator.</para>
/// <para>Layout of an image's rows (<c>clip</c>): the views' rows in raster order with <c>vision.image_newline.weight</c> [1, out] after each row of
/// positions; tiles are stitched into one grid first (a row of the grid runs through all tiles of a tile row). Tiles come first, then the global
/// view, then <c>vision.view_separator.weight</c> [1, out].</para>
/// </remarks>
public sealed record SamParameters
{
    /// <summary>Name of the graph that encodes one local tile (<see cref="TileSize"/>); the global view uses <c>vision_encode</c>.</summary>
    public const string TileGraph = "vision_encode_tile";
    /// <summary>Name of the pixel input of the view graphs, f32 [1, 3, size, size] (normalized, channels first).</summary>
    public const string PixelsInput = "pixels";
    /// <summary>Name of the newline output of the view graphs, [1, out].</summary>
    public const string NewlineOutput = "image_newline";
    /// <summary>Name of the view-separator output of the view graphs, [1, out].</summary>
    public const string SeparatorOutput = "view_separator";

    /// <summary>Backbone layers.</summary>
    public int LayerCount { get; init; }
    /// <summary>Backbone width.</summary>
    public int HiddenSize { get; init; }
    /// <summary>Backbone attention heads.</summary>
    public int HeadCount { get; init; }
    /// <summary>Backbone MLP width.</summary>
    public int FfnHiddenSize { get; init; }
    /// <summary>Epsilon of the backbone and neck LayerNorms.</summary>
    public float NormEps { get; init; } = 1e-6f;
    /// <summary>Side of the square attention windows in patches (the patch grid is zero-padded to a multiple).</summary>
    public int Window { get; init; }
    /// <summary>Layers that attend over the whole view (ascending).</summary>
    public int[] GlobalLayers { get; init; } = [];
    /// <summary>Side of the learned position grid.</summary>
    public int PositionGrid { get; init; }
    /// <summary>Channels of the convolution neck.</summary>
    public int NeckChannels { get; init; }
    /// <summary>Output channels of the stride-2 convolutions after the neck.</summary>
    public int[] DownsampleChannels { get; init; } = [];
    /// <summary>Side in pixels of the global view.</summary>
    public int GlobalSize { get; init; } = 1024;
    /// <summary>Side in pixels of a local tile.</summary>
    public int TileSize { get; init; } = 640;
    /// <summary>Fewest tiles of a tiled image.</summary>
    public int MinTiles { get; init; } = 2;
    /// <summary>Most tiles of a tiled image.</summary>
    public int MaxTiles { get; init; } = 9;
    /// <summary>8-bit value of the padding that makes the global view square.</summary>
    public int PadValue { get; init; } = 127;
    /// <summary>The second stage: <c>clip</c> (a ViT with a class token over the feature map) or <c>queries</c> (a query encoder).</summary>
    public string Encoder { get; init; } = "clip";
    /// <summary>Key-value heads of the query encoder.</summary>
    public int QueryKvHeads { get; init; }
    /// <summary>Rotary base of the query encoder.</summary>
    public float QueryRopeBase { get; init; } = 10000f;

    /// <summary>Whether the layout puts a newline row after each row of positions (the <c>clip</c> second stage).</summary>
    public bool Newlines => Encoder == "clip";

    /// <summary>Width of one backbone head.</summary>
    public int HeadDim => HiddenSize / HeadCount;

    /// <summary>Side of the feature map (after the stride-2 convolutions) of a view of <paramref name="pixels"/> pixels.</summary>
    /// <param name="pixels">Side of the view in pixels.</param>
    /// <param name="patchSize">The backbone's patch size.</param>
    public int FeatureSide(int pixels, int patchSize)
    {
        int side = pixels / patchSize;
        foreach (var _ in DownsampleChannels)
        {
            side = (side - 1) / 2 + 1;
        }

        return side;
    }

    /// <summary>Reads <c>{prefix}sam.*</c>; null when the tower has no such first stage.</summary>
    internal static SamParameters? FromMetadata(IReadOnlyDictionary<string, object> md, string prefix)
    {
        string p = prefix + "sam.";
        if (!md.ContainsKey(p + "layer_count"))
        {
            return null;
        }

        T Get<T>(string k, T d) => md.TryGetValue(p + k, out var v) ? VisionParameters.Convert<T>(v) : d;
        var s = new SamParameters
        {
            LayerCount = Get("layer_count", 0),
            HiddenSize = Get("hidden_size", 0),
            HeadCount = Get("head_count", 1),
            FfnHiddenSize = Get("ffn_hidden_size", 0),
            NormEps = Get("norm_eps", 1e-6f),
            Window = Get("window", 0),
            GlobalLayers = Get("global_layers", Array.Empty<int>()),
            PositionGrid = Get("position_grid", 0),
            NeckChannels = Get("neck_channels", 0),
            DownsampleChannels = Get("downsample_channels", Array.Empty<int>()),
            GlobalSize = Get("global_size", 1024),
            TileSize = Get("tile_size", 640),
            MinTiles = Get("min_tiles", 2),
            MaxTiles = Get("max_tiles", 9),
            PadValue = Get("pad_value", 127),
            Encoder = Get("encoder", "clip"),
            QueryKvHeads = Get("query_kv_heads", 0),
            QueryRopeBase = Get("query_rope_base", 10000f),
        };
        s.Validate();
        return s;
    }

    /// <summary>Writes the <c>sam.*</c> keys through <paramref name="set"/>.</summary>
    internal void Write(Action<string, object> set)
    {
        set("sam.layer_count", LayerCount);
        set("sam.hidden_size", HiddenSize);
        set("sam.head_count", HeadCount);
        set("sam.ffn_hidden_size", FfnHiddenSize);
        set("sam.norm_eps", NormEps);
        set("sam.window", Window);
        set("sam.global_layers", GlobalLayers);
        set("sam.position_grid", PositionGrid);
        set("sam.neck_channels", NeckChannels);
        set("sam.downsample_channels", DownsampleChannels);
        set("sam.global_size", GlobalSize);
        set("sam.tile_size", TileSize);
        set("sam.min_tiles", MinTiles);
        set("sam.max_tiles", MaxTiles);
        set("sam.pad_value", PadValue);
        if (Encoder != "clip")
        {
            set("sam.encoder", Encoder);
            set("sam.query_kv_heads", QueryKvHeads);
            set("sam.query_rope_base", QueryRopeBase);
        }
    }

    private void Validate()
    {
        if (HiddenSize <= 0 || HeadCount <= 0 || HiddenSize % HeadCount != 0 || Window <= 0 || PositionGrid <= 0 || NeckChannels <= 0
            || DownsampleChannels.Length == 0)
        {
            throw new InvalidDataException("vision.sam: hidden_size, head_count, window, position_grid, neck_channels and downsample_channels must be set consistently.");
        }

        if (GlobalLayers.Any(l => l < 0 || l >= LayerCount))
        {
            throw new InvalidDataException("vision.sam: global_layers must be backbone layer indices.");
        }

        if (Encoder is not ("clip" or "queries"))
        {
            throw new InvalidDataException($"vision.sam: encoder '{Encoder}' is not supported (clip, queries).");
        }

        if (GlobalSize <= 0 || TileSize <= 0 || MinTiles < 1 || MaxTiles < MinTiles)
        {
            throw new InvalidDataException("vision.sam: global_size, tile_size, min_tiles and max_tiles must be positive and ordered.");
        }
    }
}
