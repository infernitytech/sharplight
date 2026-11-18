using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using Sharplight.Graph;
using Sharplight.Imaging;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Mobile inverted-bottleneck image classifiers (canonical arch <c>mbconvnet</c>: MobileNetV3 and EfficientNet / EfficientNetV2 as published in timm): a strided 3x3 stem, a list of blocks
/// (<c>ir</c> inverted residual: 1x1 expansion, depthwise convolution, optional squeeze-excitation, 1x1 projection; <c>ds</c> depthwise separable; <c>er</c> fused / edge residual: k x k expansion
/// convolution, optional squeeze-excitation, 1x1 projection; <c>cn</c> a single k x k convolution) with a residual connection where the stride is 1 and the widths agree, a 1x1 head convolution, global
/// average pooling and a linear classifier. Batch norms are folded into the convolutions. Builds <see cref="WellKnown.ImageEncodeGraph"/> over the processed image as one row of <c>3 · size²</c>
/// channel-major values and describes the model as <see cref="ModelKinds.ImageClassification"/>.
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>mbconvnet.*</c>): <c>image_size</c>; <c>blocks</c> (comma separated <c>kind:stride:activation[:skip]</c> entries, activations <c>relu</c>, <c>hswish</c>, <c>silu</c>; <c>skip</c> forces
/// the residual of a <c>cn</c> block); <c>same_padding</c> (TensorFlow "same" padding of the spatial convolutions); <c>se_activation</c>, <c>se_gate</c> (<c>sigmoid</c> or <c>hsigmoid</c>);
/// <c>stem_activation</c>, <c>head_activation</c>; <c>head_pool_first</c> (MobileNetV3 pools before the head convolution); the canonical keys <see cref="CanonicalKeys.ClassifierLabels"/> (optional) and
/// <see cref="CanonicalKeys.ImageProcessor"/>. Weights: <c>stem.*</c>, <c>blocks.{i}.{exp,dw,proj}.{weight, bias}</c> (convolutions with the batch norm folded),
/// <c>blocks.{i}.se.{reduce,expand}.{weight [out, in], bias}</c>, <c>head.conv.{weight, bias}</c>, <c>classifier.{weight, bias}</c>.
/// </remarks>
public sealed class MbConvClassifierAdapter : IModelAdapter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "mbconvnet";

    /// <inheritdoc/>
    public string Name => "MobileNetV3 / EfficientNet image classifier";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => [Architecture];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        int size = src.Arch<int>("image_size");
        string[] blocks = src.Arch<string>("blocks").Split(',');
        bool same = src.Arch("same_padding", false);
        string seActivation = src.Arch("se_activation", "relu");
        string seGate = src.Arch("se_gate", "sigmoid");
        string headActivation = src.Arch("head_activation", "silu");
        bool poolFirst = src.Arch("head_pool_first", false);

        var b = new GraphBuilder(WellKnown.ImageEncodeGraph, src.Weight);
        var input = b.Input("patches", InputRole.Media, DataType.F32, new Shape(1, 3L * size * size));
        var x = Act(b, Conv(b, b.Reshape(input, 1, 3, size, size), "stem", 2, same), src.Arch("stem_activation", "silu"));
        for (int i = 0; i < blocks.Length; i++)
        {
            string[] spec = blocks[i].Split(':');
            string kind = spec[0];
            int stride = int.Parse(spec[1], CultureInfo.InvariantCulture);
            string act = spec[2];
            string p = $"blocks.{i}.";
            int channels = (int)x.Shape[1].Fixed;
            var y = x;
            switch (kind)
            {
                case "ir":
                    y = Act(b, Conv(b, y, p + "exp", 1, same), act);
                    y = Act(b, Conv(b, y, p + "dw", stride, same), act);
                    y = SqueezeExcite(b, y, p, seActivation, seGate);
                    y = Conv(b, y, p + "proj", 1, same);
                    break;
                case "ds":
                    y = Act(b, Conv(b, y, p + "dw", stride, same), act);
                    y = SqueezeExcite(b, y, p, seActivation, seGate);
                    y = Conv(b, y, p + "proj", 1, same);
                    break;
                case "er":
                    y = Act(b, Conv(b, y, p + "exp", stride, same), act);
                    y = SqueezeExcite(b, y, p, seActivation, seGate);
                    y = Conv(b, y, p + "proj", 1, same);
                    break;
                case "cn":
                    y = Act(b, Conv(b, y, p + "exp", stride, same), act);
                    break;
                default:
                    throw new InvalidOperationException($"mbconvnet: unknown block kind '{kind}'.");
            }

            bool residual = kind == "cn" ? spec.Length > 3 && spec[3] == "skip" : stride == 1;
            if (residual && y.Shape[1].Fixed == channels)
            {
                y = b.Add(x, y);
            }

            x = y;
        }

        Value pooled;
        if (poolFirst)
        {
            var avg = GlobalPool(b, x);
            pooled = Act(b, Conv(b, avg, "head.conv", 1, false), headActivation);
        }
        else
        {
            var head = Act(b, Conv(b, x, "head.conv", 1, false), headActivation);
            pooled = GlobalPool(b, head);
        }

        pooled = b.Reshape(pooled, 1, pooled.Shape[1].Fixed);
        b.Output(WellKnown.PooledOutput, pooled);
        b.Output(WellKnown.LogitsOutput, b.Linear(pooled, "classifier.weight", "classifier.bias"));

        var labels = src.Get(CanonicalKeys.ClassifierLabels, Array.Empty<string>());
        var processor = JsonNode.Parse(src.Get<string>(CanonicalKeys.ImageProcessor))!;
        var description = new ModelDescription
        {
            Kind = ModelKinds.ImageClassification,
            Config = new Dictionary<string, JsonElement>
            {
                ["labels"] = JsonSerializer.SerializeToElement(labels),
                ["scoring"] = JsonSerializer.SerializeToElement("softmax"),
            },
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
                        ["patchSize"] = JsonSerializer.SerializeToElement(size),
                        ["imageSize"] = JsonSerializer.SerializeToElement(size),
                    },
                },
            ],
        };
        return new ModelDefinition
        {
            Architecture = src.Architecture,
            Name = src.Get(CanonicalKeys.Name, src.Architecture),
            Weights = src.Weights,
            Graphs = new Dictionary<string, ComputeGraph> { [WellKnown.ImageEncodeGraph] = b.Build() },
            SourceMetadata = src.Metadata,
            OwnedResources = src.OwnedResources,
            Description = description,
        };
    }

    private static Value GlobalPool(GraphBuilder b, Value x)
    {
        long h = x.Shape[2].Fixed;
        long w = x.Shape[3].Fixed;
        return b.Emit1("Pool", [x], Attributes.From(("mode", "avg"), ("kernel", new long[] { h, w }), ("strides", new long[] { h, w })));
    }

    private static Value Act(GraphBuilder b, Value x, string activation) => activation switch
    {
        "relu" => b.Unary("Relu", x),
        "hswish" => b.Unary("HardSwish", x),
        "silu" => b.Unary("Silu", x),
        "none" => x,
        _ => throw new InvalidOperationException($"mbconvnet: unknown activation '{activation}'."),
    };

    private static Value SqueezeExcite(GraphBuilder b, Value x, string prefix, string activation, string gate)
    {
        string p = prefix + "se.";
        if (!b.HasWeight(p + "reduce.weight"))
        {
            return x;
        }

        long c = x.Shape[1].Fixed;
        var pooled = b.Reshape(GlobalPool(b, x), 1, c);
        var hidden = Act(b, b.Linear(pooled, p + "reduce.weight", p + "reduce.bias"), activation);
        var scale = b.Linear(hidden, p + "expand.weight", p + "expand.bias");
        scale = gate == "hsigmoid" ? b.Emit1("HardSigmoid", [scale], Attributes.From(("alpha", 1f / 6f), ("beta", 0.5f))) : b.Unary("Sigmoid", scale);
        return b.Mul(x, b.Reshape(scale, 1, c, 1, 1));
    }

    private static Value Conv(GraphBuilder b, Value x, string name, int stride, bool same)
    {
        var w = b.Weight(name + ".weight");
        int k = (int)w.Shape[2].Fixed;
        long h = x.Shape[2].Fixed;
        long wd = x.Shape[3].Fixed;
        long[] pads = k == 1 ? [0, 0, 0, 0] : same ? [Before(h, k, stride), Before(wd, k, stride), After(h, k, stride), After(wd, k, stride)] : [k / 2, k / 2, k / 2, k / 2];
        return b.Emit1("Conv", [x, w, b.Weight(name + ".bias")],
            Attributes.From(("strides", new long[] { stride, stride }), ("pads", pads), ("groups", x.Shape[1].Fixed / w.Shape[1].Fixed)));
    }

    private static long Total(long size, int k, int stride) => Math.Max(((((size + stride - 1) / stride) - 1) * stride) + k - size, 0);

    private static long Before(long size, int k, int stride) => Total(size, k, stride) / 2;

    private static long After(long size, int k, int stride) => Total(size, k, stride) - (Total(size, k, stride) / 2);
}
