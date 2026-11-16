using System.Text.Json;
using System.Text.Json.Nodes;
using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// DETR-style set-prediction object detectors (canonical arch <c>detr</c>): a ResNet backbone (<see cref="ResNetBackbone"/>, batch norm folded), a 1x1
/// input projection, a transformer encoder (post-norm; pre-norm with final norms for Table Transformer, <c>pre_norm</c>) over the flattened last stage with sine position embeddings, a decoder with learned object queries,
/// a class head and a three-layer box head. Builds <see cref="WellKnown.DetectGraph"/> for a fixed square input and describes the model as an
/// <see cref="ModelKinds.ObjectDetection"/> model (softmax with a no-object class) for the Core detector client.
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>detr.*</c>): <c>hidden_size</c>, <c>attention.head_count</c>, <c>encoder.layer_count</c>, <c>decoder.layer_count</c>,
/// <c>ffn_hidden_size</c>, <c>mlp.activation</c>, <c>query_count</c>, <c>image_size</c> (square input side), <c>backbone.depths</c> ("3,4,6,3"),
/// <c>backbone.bottleneck</c>; canonical <see cref="CanonicalKeys.ClassifierLabels"/> and <see cref="CanonicalKeys.ImageProcessor"/>.
/// <para>Weights: <c>backbone.*</c> (see <see cref="ResNetBackbone"/>), <c>input_proj.{weight [d, C, 1, 1],bias}</c>, <c>query_position.weight</c> [Q, d],
/// <c>encoder.layers.{i}.</c> and <c>decoder.layers.{i}.</c> with <c>self_attn.{q,k,v,o}</c> (decoder also <c>cross_attn.{q,k,v,o}</c>), the norms
/// <c>self_attn_norm</c> (<c>cross_attn_norm</c>), <c>ffn_norm</c> and <c>ffn.{up,down}</c>; <c>decoder.norm</c>; <c>class_head</c> and <c>box_head.{0,1,2}</c>.</para>
/// </remarks>
public sealed class DetrAdapter : IModelAdapter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "detr";

    /// <inheritdoc/>
    public string Name => "DETR object detector";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => [Architecture];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        int d = src.Arch<int>("hidden_size");
        int heads = src.Arch<int>("attention.head_count");
        int encoderLayers = src.Arch<int>("encoder.layer_count");
        int decoderLayers = src.Arch<int>("decoder.layer_count");
        string act = src.Arch("mlp.activation", "relu");
        int queries = src.Arch<int>("query_count");
        int size = src.Arch<int>("image_size");
        bool preNorm = src.Arch("pre_norm", false);
        var backbone = new ResNetOptions("backbone.", [.. src.Arch<string>("backbone.depths").Split(',').Select(int.Parse)], src.Arch("backbone.bottleneck", true), false, false);
        var graph = BuildGraph();
        var labels = src.Get(CanonicalKeys.ClassifierLabels, Array.Empty<string>());
        var processor = JsonNode.Parse(src.Get<string>(CanonicalKeys.ImageProcessor))!;
        var description = new ModelDescription
        {
            Kind = ModelKinds.ObjectDetection,
            Config = ComponentConfig.Of(("scoring", "softmax-background")).Concat(
                new Dictionary<string, JsonElement> { ["labels"] = JsonSerializer.SerializeToElement(labels) }).ToDictionary(),
            Components =
            [
                new ModelComponent
                {
                    Role = ModelComponentRoles.FeatureExtractor,
                    Config = new Dictionary<string, JsonElement> { [ImageProcessor.ComponentKey] = JsonSerializer.SerializeToElement(processor) },
                },
                new ModelComponent { Role = ModelComponentRoles.VisionEncoder, Graphs = [WellKnown.DetectGraph], Config = ComponentConfig.Of(("imageSize", size)) },
            ],
        };
        return new ModelDefinition
        {
            Architecture = src.Architecture,
            Name = src.Get(CanonicalKeys.Name, src.Architecture),
            Weights = src.Weights,
            Graphs = new Dictionary<string, ComputeGraph> { [WellKnown.DetectGraph] = graph },
            SourceMetadata = src.Metadata,
            OwnedResources = src.OwnedResources,
            Description = description,
        };

        ComputeGraph BuildGraph()
        {
            var b = new GraphBuilder(WellKnown.DetectGraph, src.Weight);
            var pixels = b.Input("pixels", InputRole.Media, DataType.F32, new Shape(1, 3, size, size));
            var stages = ResNetBackbone.Build(b, pixels, backbone);
            int grid = size / 32;
            int n = grid * grid;
            Value memory;
            using (b.Group("input_projection"))
            {
                var projected = b.Emit1("Conv", [stages[^1], b.Weight("input_proj.weight"), b.Weight("input_proj.bias")]);
                memory = b.Contiguous(b.Permute(b.Reshape(projected, d, n), 1, 0));
            }

            var pos = b.Constant("position_embedding", DataType.F32, new Shape(n, d), Bytes(SinePositionEmbedding.Create(grid, grid, d / 2)));
            for (int il = 0; il < encoderLayers; il++)
            {
                using var _ = b.Group($"encoder.layers.{il}");
                string p = $"encoder.layers.{il}.";
                if (preNorm)
                {
                    var normed = LayerNorm(b, memory, p + "self_attn_norm");
                    var qkn = b.Add(normed, pos);
                    memory = b.Add(memory, Attention(b, qkn, qkn, normed, p + "self_attn.", n, n));
                    memory = b.Add(memory, Ffn(b, LayerNorm(b, memory, p + "ffn_norm"), p));
                    continue;
                }

                var qk = b.Add(memory, pos);
                var attn = Attention(b, qk, qk, memory, p + "self_attn.", n, n);
                memory = LayerNorm(b, b.Add(memory, attn), p + "self_attn_norm");
                memory = LayerNorm(b, b.Add(memory, Ffn(b, memory, p)), p + "ffn_norm");
            }

            if (preNorm)
            {
                memory = LayerNorm(b, memory, "encoder.norm");
            }

            var queryPos = b.Weight("query_position.weight");
            Value h = b.Constant("decoder_input", DataType.F32, new Shape(queries, d), new byte[queries * d * 4]);
            var keys = b.Add(memory, pos);
            for (int il = 0; il < decoderLayers; il++)
            {
                using var _ = b.Group($"decoder.layers.{il}");
                string p = $"decoder.layers.{il}.";
                if (preNorm)
                {
                    var normed = LayerNorm(b, h, p + "self_attn_norm");
                    var qkn = b.Add(normed, queryPos);
                    h = b.Add(h, Attention(b, qkn, qkn, normed, p + "self_attn.", queries, queries));
                    var crossIn = LayerNorm(b, h, p + "cross_attn_norm");
                    h = b.Add(h, Attention(b, b.Add(crossIn, queryPos), keys, memory, p + "cross_attn.", queries, n));
                    h = b.Add(h, Ffn(b, LayerNorm(b, h, p + "ffn_norm"), p));
                    continue;
                }

                var qk = b.Add(h, queryPos);
                h = LayerNorm(b, b.Add(h, Attention(b, qk, qk, h, p + "self_attn.", queries, queries)), p + "self_attn_norm");
                h = LayerNorm(b, b.Add(h, Attention(b, b.Add(h, queryPos), keys, memory, p + "cross_attn.", queries, n)), p + "cross_attn_norm");
                h = LayerNorm(b, b.Add(h, Ffn(b, h, p)), p + "ffn_norm");
            }

            using (b.Group("heads"))
            {
                h = LayerNorm(b, h, "decoder.norm");
                b.Output(WellKnown.LogitsOutput, b.Linear(h, "class_head.weight", "class_head.bias"));
                var box = h;
                for (int i = 0; i < 3; i++)
                {
                    box = b.Linear(box, $"box_head.{i}.weight", $"box_head.{i}.bias");
                    if (i < 2)
                    {
                        box = b.Unary("Relu", box);
                    }
                }

                b.Output(WellKnown.BoxesOutput, b.Sigmoid(box));
            }

            return b.Build();
        }

        Value Attention(GraphBuilder b, Value query, Value key, Value value, string p, int qRows, int kRows)
        {
            int hd = d / heads;
            var q = b.Reshape(b.Linear(query, p + "q.weight", p + "q.bias"), qRows, heads, hd);
            var k = b.Reshape(b.Linear(key, p + "k.weight", p + "k.bias"), kRows, heads, hd);
            var v = b.Reshape(b.Linear(value, p + "v.weight", p + "v.bias"), kRows, heads, hd);
            var a = b.Emit1("Attention", [q, k, v], Attributes.From(("scale", 1f / MathF.Sqrt(hd)), ("causal", false)));
            return b.Linear(b.Reshape(a, qRows, d), p + "o.weight", p + "o.bias");
        }

        Value Ffn(GraphBuilder b, Value x, string p)
            => b.Linear(TowerBlocks.Activation(b, b.Linear(x, p + "ffn.up.weight", p + "ffn.up.bias"), act), p + "ffn.down.weight", p + "ffn.down.bias");

        Value LayerNorm(GraphBuilder b, Value x, string name) => TowerBlocks.LayerNorm(b, x, name, 1e-5f);

        static byte[] Bytes(float[] values) => MemoryMarshal.AsBytes(values.AsSpan()).ToArray();
    }
}
