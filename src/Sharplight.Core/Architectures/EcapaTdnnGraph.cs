using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// The ECAPA-TDNN speaker encoder (Desplanques et al. 2020, SpeechBrain's layout) as a graph over time-major log-mel frames: an initial TDNN layer, SE-Res2Net
/// blocks (a pointwise TDNN, a Res2Net block of dilated TDNNs over channel groups, a pointwise TDNN and a squeeze-excitation gate, with a residual), the multi-layer
/// feature aggregation of the blocks' outputs, attentive statistics pooling (the attention reads each frame with the recording's mean and standard deviation) and
/// a final pointwise layer. Every convolution pads "same" by reflection, as PyTorch's <c>padding="same", padding_mode="reflect"</c>.
/// </summary>
/// <remarks>
/// Weights under the prefix: <c>blocks.0.conv</c>, <c>blocks.{i}.{tdnn1,tdnn2}.conv</c>, <c>blocks.{i}.res2net_block.blocks.{j}.conv</c>,
/// <c>blocks.{i}.se_block.{conv1,conv2}</c>, <c>mfa.conv</c>, <c>asp.tdnn.conv</c>, <c>asp.conv</c> and <c>fc</c> (each <c>.weight</c> [O, C, K] and <c>.bias</c>).
/// </remarks>
internal static class EcapaTdnnGraph
{
    /// <summary>The graph name.</summary>
    public const string GraphName = "speaker_encoder";

    /// <summary>The log-mel frames [frames, melBins] (frames symbolic: <see cref="WellKnown.TokenSymbol"/>).</summary>
    public const string FeaturesInput = "features";

    /// <summary>The embedding [1, width].</summary>
    public const string EmbeddingOutput = "embedding";

    private const float StatisticsFloor = 1e-12f;

    /// <summary>Builds the graph.</summary>
    /// <param name="parameters">The encoder's hyper-parameters.</param>
    /// <param name="weights">Looks up the weights.</param>
    /// <param name="prefix">The encoder's weight prefix.</param>
    public static ComputeGraph Build(EcapaTdnnParameters parameters, Func<string, WeightInfo?> weights, string prefix)
    {
        var b = new GraphBuilder(GraphName, weights);
        var x = b.Input(FeaturesInput, InputRole.AudioFeatures, DataType.F32, new Shape(Dim.Sym(WellKnown.TokenSymbol), parameters.MelBins));
        int layers = parameters.Channels.Length;
        x = Tdnn(b, x, prefix + "blocks.0.conv.", parameters.Dilations[0], parameters.Kernels[0]);
        var outputs = new List<Value>();
        for (int i = 1; i < layers - 1; i++)
        {
            x = SeRes2NetBlock(b, x, $"{prefix}blocks.{i}.", parameters.Res2NetScale, parameters.Kernels[i], parameters.Dilations[i]);
            outputs.Add(x);
        }

        x = Tdnn(b, b.Concat(1, [.. outputs]), prefix + "mfa.conv.", parameters.Dilations[^1], parameters.Kernels[^1]);
        x = AttentiveStatistics(b, x, prefix + "asp.");
        b.Output(EmbeddingOutput, Conv(b, x, prefix + "fc.", 1, 1));
        return b.Build();
    }

    // The SE-Res2Net block: TDNN, Res2Net, TDNN, squeeze-excitation, plus the block's input.
    private static Value SeRes2NetBlock(GraphBuilder b, Value x, string n, int scale, int kernel, int dilation)
    {
        var h = Tdnn(b, x, n + "tdnn1.conv.", 1, 1);
        long width = h.Shape[1].Fixed / scale;
        var parts = new Value[scale];
        for (int i = 0; i < scale; i++)
        {
            var part = b.Slice(h, 1, i * width, width);
            parts[i] = i switch
            {
                0 => part,
                1 => Tdnn(b, part, $"{n}res2net_block.blocks.0.conv.", dilation, kernel),
                _ => Tdnn(b, b.Add(part, parts[i - 1]), $"{n}res2net_block.blocks.{i - 1}.conv.", dilation, kernel),
            };
        }

        h = Tdnn(b, b.Concat(1, parts), n + "tdnn2.conv.", 1, 1);
        var squeeze = b.Emit1("Reduce", [h], Attributes.From(("mode", "mean"), ("axes", new long[] { 0 }), ("keepdims", true)));
        squeeze = b.Unary("Relu", Conv(b, squeeze, n + "se_block.conv1.", 1, 1));
        var gate = b.Sigmoid(Conv(b, squeeze, n + "se_block.conv2.", 1, 1));
        return b.Add(b.Mul(h, gate), x);
    }

    // Attentive statistics pooling: attention weights over the frames from each frame and the recording's statistics, then the weighted mean and deviation [1, 2C].
    private static Value AttentiveStatistics(GraphBuilder b, Value x, string n)
    {
        var (mean, deviation) = Statistics(b, x, null);

        // The statistics repeated over the frames: a zero map of the frames' shape plus the row (broadcast).
        var zeros = b.Scale(x, 0f);
        var context = b.Concat(1, x, b.Add(zeros, mean), b.Add(zeros, deviation));
        var scores = Conv(b, b.Unary("Tanh", Tdnn(b, context, n + "tdnn.conv.", 1, 1)), n + "conv.", 1, 1);
        var peak = b.Emit1("Reduce", [scores], Attributes.From(("mode", "max"), ("axes", new long[] { 0 }), ("keepdims", true)));
        var exponent = b.Unary("Exp", b.Sub(scores, peak));
        var total = b.Emit1("Reduce", [exponent], Attributes.From(("mode", "sum"), ("axes", new long[] { 0 }), ("keepdims", true)));
        var (weightedMean, weightedDeviation) = Statistics(b, x, b.Binary("Div", exponent, total));
        return b.Concat(1, weightedMean, weightedDeviation);
    }

    // The mean and the standard deviation (floored at 1e-12 before the root) over the frames, uniform or under per-frame weights that sum to one.
    private static (Value Mean, Value Deviation) Statistics(GraphBuilder b, Value x, Value? weights)
    {
        Value Average(Value v) => weights is null
            ? b.Emit1("Reduce", [v], Attributes.From(("mode", "mean"), ("axes", new long[] { 0 }), ("keepdims", true)))
            : b.Emit1("Reduce", [b.Mul(v, weights)], Attributes.From(("mode", "sum"), ("axes", new long[] { 0 }), ("keepdims", true)));
        var mean = Average(x);
        var centered = b.Sub(x, mean);
        var variance = b.Emit1("Clip", [Average(b.Mul(centered, centered))], Attributes.From(("min", StatisticsFloor)));
        return (mean, b.Unary("Sqrt", variance));
    }

    private static Value Tdnn(GraphBuilder b, Value x, string n, int dilation, int kernel) => b.Unary("Relu", Conv(b, x, n, dilation, kernel));

    // A convolution with "same" reflection padding.
    private static Value Conv(GraphBuilder b, Value x, string n, int dilation, int kernel)
    {
        long padding = (long)dilation * (kernel - 1);
        long left = padding / 2;
        var attributes = padding == 0
            ? Attributes.From(("dilation", (long)dilation))
            : Attributes.From(("dilation", (long)dilation), ("pad_left", left), ("pad_right", padding - left), ("pad_mode", "reflect"));
        return b.Emit1("Conv1d", [x, b.Weight(n + "weight"), b.Weight(n + "bias")], attributes);
    }
}
