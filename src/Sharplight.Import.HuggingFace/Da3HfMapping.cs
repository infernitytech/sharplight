using System.Text.Json;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Vision;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Depth Anything 3 (<c>depth-anything/DA3-SMALL</c>, <c>DA3-BASE</c>, <c>DA3-LARGE</c>; the repositories carry the authors' <c>depth_anything_3</c> configuration, a
/// <c>config.json</c> with a <c>config</c> object of <c>net</c> / <c>head</c> / <c>cam_enc</c> / <c>cam_dec</c> entries, instead of a transformers <c>model_type</c>) → canonical
/// <c>da3_depth</c> (documented on <c>Da3DepthAdapter</c>). Single-image depth: the DualDPT head's depth branch and the backbone are mapped; the ray branch, the camera
/// encoder and decoder are not read. The position table of the 518 px training grid is resampled to the 504 px canvas the graph runs on (PyTorch's bicubic interpolation
/// with DINOv2's 0.1 offset). The metric and monocular variants (a plain DPT head with a sky branch) are not supported.
/// </summary>
internal sealed class Da3HfMapping : IHfArchitectureMapping
{
    /// <summary>The model type the configuration is recognized as.</summary>
    public const string ModelType = "depth_anything_3";

    private const int ImageSize = 504;
    private const int Patch = 14;
    private const int TrainedGrid = 37;

    private static readonly Dictionary<string, (int Hidden, int Heads, int Layers)> s_backbones = new(StringComparer.Ordinal)
    {
        ["vits"] = (384, 6, 12),
        ["vitb"] = (768, 12, 12),
        ["vitl"] = (1024, 16, 24),
    };

    /// <inheritdoc/>
    public string Architecture => "da3_depth";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = [ModelType];

    /// <inheritdoc/>
    public bool NeedsTokenizer => false;

    private static JsonElement Section(HfConfig c, string name) => c.Root.GetProperty("config").GetProperty(name);

    private static string ObjectName(JsonElement section)
        => section.TryGetProperty("__object__", out var o) && o.TryGetProperty("name", out var n) ? n.GetString() ?? "" : "";

    /// <inheritdoc/>
    public IEnumerable<string> Validate(HfConfig c)
    {
        var net = Section(c, "net");
        string backbone = net.TryGetProperty("name", out var name) ? name.GetString() ?? "" : "";
        if (!s_backbones.ContainsKey(backbone))
        {
            yield return $"depth_anything_3: the backbone '{backbone}' is not supported (vits, vitb, vitl).";
        }

        if (ObjectName(Section(c, "head")) != "DualDPT")
        {
            yield return "depth_anything_3: only the DualDPT head is supported (the metric and monocular models use a plain DPT head with a sky branch).";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HuggingFaceImportOptions options)
    {
        var net = Section(c, "net");
        var head = Section(c, "head");
        var (hidden, heads, layers) = s_backbones[net.GetProperty("name").GetString()!];
        string a = Architecture + ".";
        string Ints(JsonElement list) => string.Join(',', list.EnumerateArray().Select(v => v.GetInt32()));
        int Int(string key, int fallback) => net.TryGetProperty(key, out var v) ? v.GetInt32() : fallback;
        var processor = new ImageProcessorConfig
        {
            ResizeMode = ImageResizeMode.Exact,
            Width = ImageSize,
            Height = ImageSize,
            Filter = ImageResampleFilter.Bicubic,
            Mean = [0.485f, 0.456f, 0.406f],
            Std = [0.229f, 0.224f, 0.225f],
        };
        return new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "patch_size"] = Patch,
            [a + "image_size"] = ImageSize,
            [a + "backbone.layer_count"] = layers,
            [a + "backbone.hidden_size"] = hidden,
            [a + "backbone.head_count"] = heads,
            [a + "backbone.alt_start"] = Int("alt_start", -1),
            [a + "backbone.qknorm_start"] = Int("qknorm_start", -1),
            [a + "backbone.rope_start"] = Int("rope_start", -1),
            [a + "backbone.rope_frequency"] = 100f,
            [a + "backbone.out_layers"] = Ints(net.GetProperty("out_layers")),
            [a + "backbone.cat_token"] = !net.TryGetProperty("cat_token", out var cat) || cat.GetBoolean(),
            [a + "head.features"] = head.GetProperty("features").GetInt32(),
            [a + "head.out_channels"] = Ints(head.GetProperty("out_channels")),
            [CanonicalKeys.ImageProcessor] = processor.ToJson().ToJsonString(),
        };
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
    {
        var net = Section(c, "net");
        int hidden = s_backbones[net.GetProperty("name").GetString()!].Hidden;
        int grid = ImageSize / Patch;
        var collector = new HfWeightCollector();
        foreach (var (rawName, tensor) in tensors)
        {
            string name = rawName.StartsWith("model.", StringComparison.Ordinal) ? rawName["model.".Length..] : rawName;
            var info = tensor.ToWeightInfo();
            if (name.StartsWith("cam_enc.", StringComparison.Ordinal) || name.StartsWith("cam_dec.", StringComparison.Ordinal) || name.Contains("_aux", StringComparison.Ordinal))
            {
                continue;
            }

            switch (name)
            {
                case "backbone.pretrained.cls_token":
                    collector.Add(HfTransforms.ToF32(info, "backbone.cls_token", new Tensors.Shape(1, hidden)));
                    continue;
                case "backbone.pretrained.camera_token":
                    // [1, 2, hidden]: the reference view's token (row 0) replaces the class token; the second is for the other views.
                    collector.Add(HfTransforms.ToF32(info.As("backbone.camera_token", new Tensors.Shape(2, hidden)).RowSlice("backbone.camera_token", 0, 1), "backbone.camera_token"));
                    continue;
                case "backbone.pretrained.pos_embed":
                    collector.Add(PositionEmbedding(info, hidden, grid));
                    continue;
                case "backbone.pretrained.patch_embed.proj.weight":
                    collector.Add(HfTransforms.Plain(info, "backbone.patch_embed.weight").As("backbone.patch_embed.weight", new Tensors.Shape(info.Shape[0].Fixed, info.Shape.Elements() / info.Shape[0].Fixed)));
                    continue;
            }

            string? canonical = name.StartsWith("backbone.pretrained.", StringComparison.Ordinal)
                ? "backbone." + name["backbone.pretrained.".Length..].Replace("patch_embed.proj.", "patch_embed.", StringComparison.Ordinal)
                : name.StartsWith("head.", StringComparison.Ordinal) ? name : null;
            if (canonical is null)
            {
                collector.AddUnmapped(rawName);
                continue;
            }

            collector.Add(info.Shape.Rank == 1 ? HfTransforms.Plain(info, canonical) : info.As(canonical));
        }

        collector.ThrowIfUnmapped(c.ModelType);
        return collector.Weights;
    }

    // [1, 1 + 37², hidden] → [1 + grid², hidden]: the class row is kept, the patch grid is resampled as DINOv2 does for another input size.
    private static WeightInfo PositionEmbedding(WeightInfo info, int hidden, int grid)
    {
        long rows = 1 + ((long)TrainedGrid * TrainedGrid);
        if (info.Shape.Elements() != rows * hidden)
        {
            throw new InvalidDataException($"{info.Name}: expected a position table for a {TrainedGrid}x{TrainedGrid} grid.");
        }

        return WeightInfo.Derived("backbone.position_embedding.weight", Tensors.DataType.F32, new Tensors.Shape(1 + ((long)grid * grid), hidden), () =>
        {
            var values = HfTransforms.DecodeF32(info);
            double sourcePerTarget = TrainedGrid / (grid + 0.1);   // DINOv2 passes scale_factor = (grid + 0.1) / 37 to interpolate
            var patches = grid == TrainedGrid
                ? values.AsSpan(hidden).ToArray()
                : PositionGridResampling.ResizeGrid(values.AsSpan(hidden), TrainedGrid, grid, hidden, antialias: false, sourcePerTarget);
            var all = new float[hidden + patches.Length];
            values.AsSpan(0, hidden).CopyTo(all);
            patches.CopyTo(all.AsSpan(hidden));
            return HfTransforms.EncodeF32(all);
        });
    }
}
