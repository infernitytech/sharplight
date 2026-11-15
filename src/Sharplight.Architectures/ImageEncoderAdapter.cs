using System.Text.Json;
using System.Text.Json.Nodes;
using Sharplight.Graph;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Plain vision transformer image encoders (canonical arch <c>vit</c>): ViT / DeiT classifiers, DINOv2 (with LayerScale and register
/// tokens) and the like. Builds <see cref="WellKnown.ImageEncodeGraph"/> over the host-patchified image and describes the model as an
/// <see cref="ModelKinds.ImageClassification"/> (classifier head present) or <see cref="ModelKinds.ImageEmbedding"/> model with a
/// <see cref="ModelComponentRoles.FeatureExtractor"/> (the stored <see cref="ImageProcessorConfig"/>) and a
/// <see cref="ModelComponentRoles.VisionEncoder"/> component; the Core image clients run it.
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>vit.*</c>): <c>layer_count</c>, <c>hidden_size</c>, <c>ffn_hidden_size</c>, <c>attention.head_count</c>, <c>norm_eps</c>,
/// <c>mlp.activation</c> ("gelu" (erf, default), "gelu_tanh", "quick_gelu", "relu", "silu"), <c>patch_size</c>, <c>image_size</c> (the side of
/// the square input; the position table has exactly the grid of this size), <c>register_count</c> (default 0), <c>pooling</c> ("cls" (default)
/// or "mean": the mean of the patch rows). The canonical keys <see cref="CanonicalKeys.ClassifierLabels"/>, <see cref="CanonicalKeys.ClassifierActivation"/>
/// and <see cref="CanonicalKeys.ImageProcessor"/> describe the model's classes and preprocessing.
/// <para>Weights: <c>patch_embed.weight</c> [hidden, 3·patch²] (the patch convolution flattened channel-major) and <c>.bias</c>;
/// <c>class_embedding</c> [hidden]; optional <c>register_embedding.weight</c> [registers, hidden]; <c>position_embedding.weight</c>
/// [1 + grid², hidden] (row 0 is the class token's); per layer <c>layers.{i}.</c>: <c>input_norm</c>, <c>attention.{q,k,v,o}</c> (+ biases),
/// optional <c>layer_scale_1.weight</c> [hidden], <c>post_attention_norm</c>, <c>mlp.{up,down}</c> (+ biases), optional <c>layer_scale_2.weight</c>;
/// <c>output_norm</c> (applied to every row before pooling); optional <c>classifier.out.{weight,bias}</c> [labels, hidden]. Norms are LayerNorms.</para>
/// </remarks>
public sealed class ImageEncoderAdapter : IModelAdapter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "vit";

    /// <inheritdoc/>
    public string Name => "Vision transformer image encoder";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => [Architecture];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        int layers = src.Arch<int>("layer_count");
        int hidden = src.Arch<int>("hidden_size");
        int heads = src.Arch<int>("attention.head_count");
        int patch = src.Arch<int>("patch_size");
        int imageSize = src.Arch<int>("image_size");
        int registers = src.Arch("register_count", 0);
        float eps = src.Arch("norm_eps", 1e-6f);
        string act = src.Arch("mlp.activation", "gelu");
        string pooling = src.Arch("pooling", "cls");
        int head = hidden / heads;
        int grid = imageSize / patch;
        bool classifier = src.Weight("classifier.out.weight") is not null;
        var graph = BuildGraph();

        var config = new Dictionary<string, JsonElement>();
        if (classifier)
        {
            var labels = src.Get(CanonicalKeys.ClassifierLabels, Array.Empty<string>());
            config["labels"] = JsonSerializer.SerializeToElement(labels);
            config["scoring"] = JsonSerializer.SerializeToElement(src.Get(CanonicalKeys.ClassifierActivation, "softmax"));
        }

        var processor = JsonNode.Parse(src.Get<string>(CanonicalKeys.ImageProcessor))!;
        var description = new ModelDescription
        {
            Kind = classifier ? ModelKinds.ImageClassification : ModelKinds.ImageEmbedding,
            Config = config,
            Components =
            [
                new ModelComponent
                {
                    Role = ModelComponentRoles.FeatureExtractor,
                    Config = new Dictionary<string, JsonElement> { [ImageProcessor.ComponentKey] = JsonSerializer.SerializeToElement(processor) },
                },
                new ModelComponent
                {
                    Role = ModelComponentRoles.VisionEncoder,
                    Graphs = [WellKnown.ImageEncodeGraph],
                    Config = new Dictionary<string, JsonElement>
                    {
                        ["patchSize"] = JsonSerializer.SerializeToElement(patch),
                        ["imageSize"] = JsonSerializer.SerializeToElement(imageSize),
                    },
                },
            ],
        };
        return new ModelDefinition
        {
            Architecture = src.Architecture,
            Name = src.Get(CanonicalKeys.Name, src.Architecture),
            Weights = src.Weights,
            Graphs = new Dictionary<string, ComputeGraph> { [WellKnown.ImageEncodeGraph] = graph },
            SourceMetadata = src.Metadata,
            OwnedResources = src.OwnedResources,
            Description = description,
        };

        ComputeGraph BuildGraph()
        {
            var b = new GraphBuilder(WellKnown.ImageEncodeGraph, src.Weight);
            int n = grid * grid;
            var patches = b.Input("patches", InputRole.Media, DataType.F32, new Shape(n, 3 * patch * patch));
            Value x;
            using (b.Group("embedding"))
            {
                x = b.Linear(patches, "patch_embed.weight", "patch_embed.bias", "patch_embd");
                var cls = b.Reshape(b.Weight("class_embedding"), 1, hidden);
                x = b.Concat(0, cls, x);
                x = b.Add(x, b.Weight("position_embedding.weight"), "inp_pos_embd");
                if (registers > 0)
                {
                    x = b.Concat(0, b.Slice(x, 0, 0, 1), b.Weight("register_embedding.weight"), b.Slice(x, 0, 1, n));
                }
            }

            int rows = 1 + registers + n;
            for (int il = 0; il < layers; il++)
            {
                using var _ = b.Group($"layers.{il}");
                string p = $"layers.{il}.";
                var h = LayerNorm(b, x, p + "input_norm");
                var q = b.Reshape(b.Linear(h, p + "attention.q.weight", p + "attention.q.bias"), rows, heads, head);
                var k = b.Reshape(b.Linear(h, p + "attention.k.weight", p + "attention.k.bias"), rows, heads, head);
                var v = b.Reshape(b.Linear(h, p + "attention.v.weight", p + "attention.v.bias"), rows, heads, head);
                var attn = b.Emit1("Attention", [q, k, v], Attributes.From(("scale", 1f / MathF.Sqrt(head)), ("causal", false)), $"{p}attn");
                var o = b.Linear(b.Reshape(attn, rows, hidden), p + "attention.o.weight", p + "attention.o.bias");
                x = b.Add(x, Scaled(b, o, p + "layer_scale_1.weight"), $"{p}attn_residual");
                h = LayerNorm(b, x, p + "post_attention_norm");
                var mlp = b.Linear(Activation(b, b.Linear(h, p + "mlp.up.weight", p + "mlp.up.bias")), p + "mlp.down.weight", p + "mlp.down.bias");
                x = b.Add(x, Scaled(b, mlp, p + "layer_scale_2.weight"), $"{p}l_out");
            }

            using (b.Group("output"))
            {
                x = LayerNorm(b, x, "output_norm");
                Value pooled;
                if (pooling == "mean")
                {
                    var patchRows = b.Slice(x, 0, 1 + registers, n);
                    var sum = b.Emit1("ReduceSum", [b.Contiguous(b.Permute(patchRows, 1, 0))]);
                    pooled = b.Reshape(b.Scale(sum, 1f / n), 1, hidden);
                }
                else
                {
                    pooled = b.Slice(x, 0, 0, 1);
                }

                b.Output(WellKnown.PooledOutput, pooled);
                if (classifier)
                {
                    b.Output(WellKnown.LogitsOutput, b.Linear(pooled, "classifier.out.weight", "classifier.out.bias", "logits"));
                }
            }

            return b.Build();
        }

        Value LayerNorm(GraphBuilder b, Value v, string name)
            => b.Emit1("LayerNorm", [v, b.Weight(name + ".weight"), b.Weight(name + ".bias")], Attributes.From(("eps", eps)));

        Value Scaled(GraphBuilder b, Value v, string scale) => b.HasWeight(scale) ? b.Mul(v, b.Weight(scale)) : v;

        Value Activation(GraphBuilder b, Value t) => act switch
        {
            "gelu" => b.Unary("Gelu", t),
            "gelu_tanh" => b.Unary("GeluTanh", t),
            "quick_gelu" => b.Mul(t, b.Sigmoid(b.Scale(t, 1.702f))),
            "relu" => b.Unary("Relu", t),
            "silu" => b.Silu(t),
            _ => throw new NotSupportedException($"vit: unknown mlp.activation '{act}'."),
        };
    }
}
