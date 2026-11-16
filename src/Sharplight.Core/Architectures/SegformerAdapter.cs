using System.Text.Json;
using System.Text.Json.Nodes;
using Sharplight.Graph;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// SegFormer semantic segmentation (canonical arch <c>segformer</c>): a mix-transformer encoder of four stages (overlapping patch-embedding convolutions, blocks of
/// efficient self-attention whose keys and values come from a strided spatial-reduction convolution of the stage map, and Mix-FFN blocks with a depthwise 3x3
/// convolution) and the all-MLP decoder (a linear projection per stage, bilinear resampling to the first stage's resolution, concatenation, a 1x1 fusion convolution,
/// ReLU and the classifier). Builds <see cref="WellKnown.SemanticSegmentGraph"/> for a fixed square input and describes the model as <see cref="ModelKinds.SemanticSegmentation"/>.
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>segformer.*</c>): <c>image_size</c>, <c>depths</c>, <c>hidden_sizes</c>, <c>head_counts</c>, <c>sr_ratios</c>, <c>patch_sizes</c>, <c>strides</c>
/// (comma-separated per stage), <c>mlp_ratio</c> (per stage list), <c>decoder_hidden_size</c>, <c>norm_eps</c>, <c>mlp.activation</c>; canonical keys
/// <see cref="CanonicalKeys.ClassifierLabels"/> and <see cref="CanonicalKeys.ImageProcessor"/>. Weights: <c>stages.{i}.embed</c> (convolution), <c>stages.{i}.embed_norm</c>,
/// <c>stages.{i}.blocks.{j}.{norm1, attn.q, attn.k, attn.v, attn.o, attn.sr (convolution, absent when the ratio is 1), attn.sr_norm, norm2, mlp.up, mlp.dw (depthwise convolution),
/// mlp.down}</c>, <c>stages.{i}.norm</c>, <c>decode.proj.{i}</c>, <c>decode.fuse</c> (convolution with the batch norm folded in), <c>decode.classifier</c> (convolution).
/// </remarks>
public sealed class SegformerAdapter : IModelAdapter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "segformer";

    /// <inheritdoc/>
    public string Name => "SegFormer semantic segmenter";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => [Architecture];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        int size = src.Arch<int>("image_size");
        var graph = BuildGraph(src, size);
        var labels = src.Get(CanonicalKeys.ClassifierLabels, Array.Empty<string>());
        var processor = JsonNode.Parse(src.Get<string>(CanonicalKeys.ImageProcessor))!;
        var description = new ModelDescription
        {
            Kind = ModelKinds.SemanticSegmentation,
            Config = new Dictionary<string, JsonElement> { ["labels"] = JsonSerializer.SerializeToElement(labels) },
            Components =
            [
                new ModelComponent
                {
                    Role = ModelComponentRoles.FeatureExtractor,
                    Config = new Dictionary<string, JsonElement> { [ImageProcessor.ComponentKey] = JsonSerializer.SerializeToElement(processor) },
                },
                new ModelComponent { Role = ModelComponentRoles.VisionEncoder, Graphs = [WellKnown.SemanticSegmentGraph], Config = ComponentConfig.Of(("imageSize", size)) },
            ],
        };
        return new ModelDefinition
        {
            Architecture = src.Architecture,
            Name = src.Get(CanonicalKeys.Name, src.Architecture),
            Weights = src.Weights,
            Graphs = new Dictionary<string, ComputeGraph> { [WellKnown.SemanticSegmentGraph] = graph },
            SourceMetadata = src.Metadata,
            OwnedResources = src.OwnedResources,
            Description = description,
        };
    }

    private static int[] Ints(ModelSource src, string key) => [.. src.Arch<string>(key).Split(',').Select(int.Parse)];

    private static ComputeGraph BuildGraph(ModelSource src, int size)
    {
        int[] depths = Ints(src, "depths");
        int[] hidden = Ints(src, "hidden_sizes");
        int[] heads = Ints(src, "head_counts");
        int[] reductions = Ints(src, "sr_ratios");
        int[] patches = Ints(src, "patch_sizes");
        int[] strides = Ints(src, "strides");
        int[] mlpRatios = Ints(src, "mlp_ratios");
        int decoder = src.Arch<int>("decoder_hidden_size");
        string activation = src.Arch("mlp.activation", "gelu");
        const float eps = 1e-5f;

        var b = new GraphBuilder(WellKnown.SemanticSegmentGraph, src.Weight);
        var pixels = b.Input("pixels", InputRole.Media, DataType.F32, new Shape(1, 3, size, size));
        var map = pixels;
        int h = size;
        int w = size;
        var rowsPerStage = new Value[depths.Length];
        var sizes = new (int H, int W)[depths.Length];
        for (int s = 0; s < depths.Length; s++)
        {
            using var stage = b.Group($"stages.{s}");
            int c = hidden[s];
            int pad = patches[s] / 2;
            map = Conv(b, map, $"stages.{s}.embed", strides[s], pad, 1);
            h = (h + 2 * pad - patches[s]) / strides[s] + 1;
            w = (w + 2 * pad - patches[s]) / strides[s] + 1;
            var x = TowerBlocks.LayerNorm(b, ToRows(b, map, c, h * w), $"stages.{s}.embed_norm", eps);
            for (int j = 0; j < depths[s]; j++)
            {
                using var block = b.Group($"stages.{s}.blocks.{j}");
                string p = $"stages.{s}.blocks.{j}.";
                var normed = TowerBlocks.LayerNorm(b, x, p + "norm1", eps);
                x = b.Add(x, Attention(b, normed, p + "attn.", c, heads[s], reductions[s], h, w, eps));
                var m = TowerBlocks.LayerNorm(b, x, p + "norm2", eps);
                int inner = c * mlpRatios[s];
                var up = ToMap(b, b.Linear(m, p + "mlp.up.weight", p + "mlp.up.bias"), inner, h, w);
                var dw = Conv(b, up, p + "mlp.dw", 1, 1, inner);
                x = b.Add(x, b.Linear(TowerBlocks.Activation(b, ToRows(b, dw, inner, h * w), activation), p + "mlp.down.weight", p + "mlp.down.bias"));
            }

            x = TowerBlocks.LayerNorm(b, x, $"stages.{s}.norm", eps);
            rowsPerStage[s] = x;
            sizes[s] = (h, w);
            map = ToMap(b, x, c, h, w);
        }

        using (b.Group("decode"))
        {
            var fused = new Value[depths.Length];
            for (int s = 0; s < depths.Length; s++)
            {
                var projected = ToMap(b, b.Linear(rowsPerStage[s], $"decode.proj.{s}.weight", $"decode.proj.{s}.bias"), decoder, sizes[s].H, sizes[s].W);
                if (s > 0)
                {
                    projected = b.Emit1("Resize", [projected], Attributes.From(("size", new long[] { sizes[0].H, sizes[0].W }), ("mode", "linear"), ("coord", "half_pixel")));
                }

                fused[depths.Length - 1 - s] = projected;
            }

            var cat = b.Concat(1, fused);
            var y = b.Unary("Relu", Conv(b, cat, "decode.fuse", 1, 0, 1));
            b.Output(WellKnown.SemanticLogitsOutput, Conv(b, y, "decode.classifier", 1, 0, 1));
        }

        return b.Build();
    }

    // Efficient self-attention: queries from every row, keys and values from the (spatially reduced) rows.
    private static Value Attention(GraphBuilder b, Value x, string p, int c, int heads, int reduction, int h, int w, float eps)
    {
        int hd = c / heads;
        int rows = h * w;
        var kv = x;
        int kvRows = rows;
        if (reduction > 1)
        {
            var reduced = Conv(b, ToMap(b, x, c, h, w), p + "sr", reduction, 0, 1);
            kvRows = (h / reduction) * (w / reduction);
            kv = TowerBlocks.LayerNorm(b, ToRows(b, reduced, c, kvRows), p + "sr_norm", eps);
        }

        var q = b.Reshape(b.Linear(x, p + "q.weight", p + "q.bias"), rows, heads, hd);
        var k = b.Reshape(b.Linear(kv, p + "k.weight", p + "k.bias"), kvRows, heads, hd);
        var v = b.Reshape(b.Linear(kv, p + "v.weight", p + "v.bias"), kvRows, heads, hd);
        var a = b.Emit1("Attention", [q, k, v], Attributes.From(("scale", 1f / MathF.Sqrt(hd)), ("causal", false)));
        return b.Linear(b.Reshape(a, rows, c), p + "o.weight", p + "o.bias");
    }

    private static Value ToRows(GraphBuilder b, Value map, int channels, int rows)
        => b.Contiguous(b.Permute(b.Reshape(map, channels, rows), 1, 0));

    private static Value ToMap(GraphBuilder b, Value rows, int channels, int h, int w)
        => b.Reshape(b.Contiguous(b.Permute(rows, 1, 0)), 1, channels, h, w);

    private static Value Conv(GraphBuilder b, Value x, string name, int stride, int pad, int groups)
    {
        var inputs = b.HasWeight(name + ".bias") ? new[] { x, b.Weight(name + ".weight"), b.Weight(name + ".bias") } : [x, b.Weight(name + ".weight")];
        return b.Emit1("Conv", inputs, Attributes.From(("strides", new long[] { stride, stride }), ("pads", new long[] { pad, pad, pad, pad }), ("groups", (long)groups)));
    }
}
