using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// The CAM++ speaker encoder (3D-Speaker's <c>CAMPPlus</c>, the x-vector of CosyVoice and Chatterbox) as a graph over time-major Kaldi filter-bank
/// frames: the FCM head (2-D residual convolutions over frequency and time, frequency strided), a strided TDNN layer, three densely connected
/// TDNN blocks of context-aware masking layers (each mask from the mean over the recording plus the mean over its 100-frame segment) with
/// transition layers, statistics pooling (mean and unbiased standard deviation over time) and the dense embedding layer. Batch norms use their
/// running statistics. The host supplies the segments of the frames after the strided layer and the unbiased-variance factor. An export that folded a
/// batch norm into the convolution before it (CosyVoice's <c>campplus.onnx</c>) gives that convolution a bias and omits the norm's statistics: the
/// convolution then adds its bias and the norm is skipped.
/// </summary>
internal static class CamPlusGraph
{
    /// <summary>The graph name.</summary>
    public const string GraphName = "campplus";

    /// <summary>The filter-bank frames [frames, melBins] (frames symbolic: <see cref="WellKnown.TokenSymbol"/>).</summary>
    public const string FeaturesInput = "features";

    /// <summary>The 100-frame segments of the strided frames: i32 [segments, 2] (first frame, frame count).</summary>
    public const string SegmentsInput = "segments";

    /// <summary>The segment of every strided frame: i32 [strided frames].</summary>
    public const string SegmentIndexInput = "segment_index";

    /// <summary>n / (n − 1) for the n strided frames: f32 [1].</summary>
    public const string VarianceFactorInput = "variance_factor";

    /// <summary>The embedding [1, width].</summary>
    public const string EmbeddingOutput = "embedding";

    /// <summary>The symbol of the segment count.</summary>
    public const string SegmentSymbol = "segments";

    /// <summary>The frames per pooling segment of the masking layers.</summary>
    public const int SegmentLength = 100;

    private const float Eps = 1e-5f;

    /// <summary>Builds the graph.</summary>
    /// <param name="weights">Looks up the weights.</param>
    /// <param name="prefix">The encoder's weight prefix (<c>s3gen.speaker_encoder.</c>).</param>
    /// <param name="melBins">The filter-bank channels.</param>
    public static ComputeGraph Build(Func<string, WeightInfo?> weights, string prefix, int melBins)
    {
        var b = new GraphBuilder(GraphName, weights);
        var features = b.Input(FeaturesInput, InputRole.AudioFeatures, DataType.F32, new Shape(Dim.Sym(WellKnown.TokenSymbol), melBins));
        var x = Head(b, b.Reshape(features, 0, 1, melBins), prefix + "head.");
        string v = prefix + "xvector.";
        x = b.Emit1("Conv1d", [x, .. WithBias(b, v + "tdnn.linear.")], Attributes.From(("stride", 2L), ("pad_left", 2L), ("pad_right", 2L)));
        x = BatchNormRelu(b, x, v + "tdnn.nonlinear.batchnorm.");
        var segments = b.Input(SegmentsInput, InputRole.Other, DataType.I32, new Shape(Dim.Sym(SegmentSymbol), 2));
        var segmentIndex = b.Input(SegmentIndexInput, InputRole.Other, DataType.I32, new Shape(x.Shape[0]));
        (int Layers, long Dilation)[] blocks = [(12, 1), (24, 2), (16, 2)];
        for (int i = 0; i < blocks.Length; i++)
        {
            for (int l = 0; l < blocks[i].Layers; l++)
            {
                string n = $"{v}block{i + 1}.tdnnd{l + 1}.";
                var h = Pointwise(b, BatchNormRelu(b, x, n + "nonlinear1.batchnorm."), n + "linear1.weight");
                h = BatchNormRelu(b, h, n + "nonlinear2.batchnorm.");
                x = b.Concat(1, x, ContextMask(b, h, n + "cam_layer.", blocks[i].Dilation, segments, segmentIndex));
            }

            string t = $"{v}transit{i + 1}.";
            x = Pointwise(b, BatchNormRelu(b, x, t + "nonlinear.batchnorm."), t + "linear.weight");
        }

        x = BatchNormRelu(b, x, v + "out_nonlinear.batchnorm.");
        var mean = b.Emit1("Reduce", [x], Attributes.From(("mode", "mean"), ("axes", new long[] { 0 }), ("keepdims", true)));
        var centered = b.Sub(x, mean);
        var variance = b.Emit1("Reduce", [b.Mul(centered, centered)], Attributes.From(("mode", "mean"), ("axes", new long[] { 0 }), ("keepdims", true)));
        var factor = b.Input(VarianceFactorInput, InputRole.Other, DataType.F32, new Shape(1));
        var deviation = b.Unary("Sqrt", b.Mul(variance, factor));
        var stats = b.Concat(1, mean, deviation);
        var embedding = Pointwise(b, stats, v + "dense.linear.weight");
        b.Output(EmbeddingOutput, BatchNorm(b, embedding, v + "dense.nonlinear.batchnorm.", affine: false));
        return b.Build();
    }

    // FCM over [frames, 1, melBins]: conv, two residual stages and a strided conv over frequency, then [frames, channels · melBins / 8].
    private static Value Head(GraphBuilder b, Value x, string h)
    {
        x = Relu(b, BatchNorm2d(b, Conv2d(b, x, h + "conv1.weight", 1), h + "bn1."));
        for (int stage = 1; stage <= 2; stage++)
        {
            for (int block = 0; block < 2; block++)
            {
                string n = $"{h}layer{stage}.{block}.";
                long stride = block == 0 ? 2 : 1;
                var y = Relu(b, BatchNorm2d(b, Conv2d(b, x, n + "conv1.weight", stride), n + "bn1."));
                y = BatchNorm2d(b, Conv2d(b, y, n + "conv2.weight", 1), n + "bn2.");
                var shortcut = b.HasWeight(n + "shortcut.0.weight")
                    ? BatchNorm2d(b, Conv2d(b, x, n + "shortcut.0.weight", stride, padding: 0), n + "shortcut.1.")
                    : x;
                x = Relu(b, b.Add(y, shortcut));
            }
        }

        x = Relu(b, BatchNorm2d(b, Conv2d(b, x, h + "conv2.weight", 2), h + "bn2."));
        long channels = x.Shape[1].Fixed * x.Shape[2].Fixed;
        return b.Reshape(x, 0, channels);
    }

    // A PyTorch Conv2d over (frequency, time) on the time-major map [frames, C, frequency]: the kernel's axes swapped, strided over frequency only.
    private static Value Conv2d(GraphBuilder b, Value x, string weight, long frequencyStride, long padding = 1)
    {
        var kernel = b.Permute(b.Weight(weight), 0, 1, 3, 2);
        var attributes = Attributes.From(("strides", new[] { 1L, frequencyStride }), ("pads", new[] { padding, padding, padding, padding }));
        string bias = weight[..^"weight".Length] + "bias";
        Value[] inputs = b.HasWeight(bias) ? [x, kernel, b.Weight(bias)] : [x, kernel];
        return b.Emit1("SequenceConv2d", inputs, attributes);
    }

    // The context-aware masking layer: the local convolution scaled by a sigmoid mask from the recording's and the segment's mean context.
    private static Value ContextMask(GraphBuilder b, Value x, string n, long dilation, Value segments, Value segmentIndex)
    {
        var local = b.Emit1("Conv1d", [x, b.Weight(n + "linear_local.weight")], Attributes.From(("dilation", dilation), ("pad_left", dilation), ("pad_right", dilation)));
        var global = b.Emit1("Reduce", [x], Attributes.From(("mode", "mean"), ("axes", new long[] { 0 }), ("keepdims", true)));
        var segment = b.Gather(b.Emit1("SequencePool", [x, segments], Attributes.From(("mode", "mean"))), segmentIndex);
        var context = b.Add(global, segment);
        context = b.Unary("Relu", Pointwise(b, context, n + "linear1.weight", n + "linear1.bias"));
        var mask = b.Sigmoid(Pointwise(b, context, n + "linear2.weight", n + "linear2.bias"));
        return b.Mul(local, mask);
    }

    // A kernel-1 Conv1d as a linear layer over the rows (with the bias of a folded batch norm when there is one).
    private static Value Pointwise(GraphBuilder b, Value x, string weight, string? bias = null)
    {
        bias ??= b.HasWeight(weight[..^"weight".Length] + "bias") ? weight[..^"weight".Length] + "bias" : null;
        Value[] inputs = bias is null ? [x, b.Weight(weight)] : [x, b.Weight(weight), b.Weight(bias)];
        return b.Emit1("Conv1d", inputs, Attributes.Empty);
    }

    // A convolution's weight and, when it has one, its bias.
    private static Value[] WithBias(GraphBuilder b, string module)
        => b.HasWeight(module + "bias") ? [b.Weight(module + "weight"), b.Weight(module + "bias")] : [b.Weight(module + "weight")];

    private static Value BatchNormRelu(GraphBuilder b, Value x, string n) => b.Unary("Relu", BatchNorm(b, x, n, affine: true));

    private static Value Relu(GraphBuilder b, Value x) => b.Unary("Relu", x);

    // Inference batch norm over the last axis of [rows, C]: (x − mean) / sqrt(var + eps) · weight + bias.
    private static Value BatchNorm(GraphBuilder b, Value x, string n, bool affine)
    {
        if (!b.HasWeight(n + "running_var"))
        {
            return x;
        }

        var inverse = b.Unary("Rsqrt", b.Scale(b.Weight(n + "running_var"), 1f, Eps));
        var y = b.Mul(b.Sub(x, b.Weight(n + "running_mean")), inverse);
        return affine ? b.Add(b.Mul(y, b.Weight(n + "weight")), b.Weight(n + "bias")) : y;
    }

    // Inference batch norm over axis 1 of [frames, C, frequency].
    private static Value BatchNorm2d(GraphBuilder b, Value x, string n)
    {
        if (!b.HasWeight(n + "running_var"))
        {
            return x;
        }

        long channels = x.Shape[1].Fixed;
        Value Column(Value v) => b.Reshape(v, channels, 1);
        var inverse = b.Unary("Rsqrt", b.Scale(Column(b.Weight(n + "running_var")), 1f, Eps));
        var y = b.Mul(b.Sub(x, Column(b.Weight(n + "running_mean"))), inverse);
        return b.Add(b.Mul(y, Column(b.Weight(n + "weight"))), Column(b.Weight(n + "bias")));
    }
}
