using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// XTTS v2's speaker encoder (Coqui's <c>ResNetSpeakerEncoder</c> in the HiFi-GAN decoder) as a graph over time-major log-mel frames: instance
/// normalization per channel, a convolution, four stages of squeeze-and-excitation residual blocks (2-D over frequency and time; the later stages
/// strided by two in both), attentive statistics pooling (a softmax over time per channel, the weighted mean and standard deviation) and the
/// projection. The host L2-normalizes the embedding.
/// </summary>
internal static class XttsSpeakerEncoderGraph
{
    /// <summary>The graph name.</summary>
    public const string GraphName = "xtts_speaker_encoder";

    /// <summary>The log-mel frames [frames, 64] (frames symbolic: <see cref="WellKnown.TokenSymbol"/>).</summary>
    public const string MelInput = "mel";

    /// <summary>The unnormalized embedding [1, width].</summary>
    public const string EmbeddingOutput = "embedding";

    private const string P = "hifigan_decoder.speaker_encoder.";
    private const float Eps = 1e-5f;

    /// <summary>Builds the graph.</summary>
    /// <param name="weights">Looks up the weights.</param>
    /// <param name="melBins">The mel channels.</param>
    public static ComputeGraph Build(Func<string, WeightInfo?> weights, int melBins)
    {
        var b = new GraphBuilder(GraphName, weights);
        var mel = b.Input(MelInput, InputRole.AudioFeatures, DataType.F32, new Shape(Dim.Sym(WellKnown.TokenSymbol), melBins));
        var mean = b.Emit1("Reduce", [mel], Attributes.From(("mode", "mean"), ("axes", new long[] { 0 }), ("keepdims", true)));
        var centered = b.Sub(mel, mean);
        var variance = b.Emit1("Reduce", [b.Mul(centered, centered)], Attributes.From(("mode", "mean"), ("axes", new long[] { 0 }), ("keepdims", true)));
        var x = b.Reshape(b.Mul(centered, b.Unary("Rsqrt", b.Scale(variance, 1f, Eps))), 0, 1, melBins);

        x = BatchNorm2d(b, b.Unary("Relu", Conv2d(b, x, P + "conv1", 1, bias: true)), P + "bn1.");
        (int Blocks, long Stride)[] stages = [(3, 1), (4, 2), (6, 2), (3, 2)];
        for (int s = 0; s < stages.Length; s++)
        {
            for (int i = 0; i < stages[s].Blocks; i++)
            {
                string n = $"{P}layer{s + 1}.{i}.";
                long stride = i == 0 ? stages[s].Stride : 1;
                var y = BatchNorm2d(b, b.Unary("Relu", Conv2d(b, x, n + "conv1", stride)), n + "bn1.");
                y = BatchNorm2d(b, Conv2d(b, y, n + "conv2", 1), n + "bn2.");
                y = SqueezeExcite(b, y, n + "se.fc.");
                var residual = b.HasWeight(n + "downsample.0.weight") ? BatchNorm2d(b, Conv2d(b, x, n + "downsample.0", stride, padding: 0), n + "downsample.1.") : x;
                x = b.Unary("Relu", b.Add(y, residual));
            }
        }

        long channels = x.Shape[1].Fixed * x.Shape[2].Fixed;
        x = b.Reshape(x, 0, channels);
        var w = Pointwise(b, x, P + "attention.0");
        w = BatchNorm(b, b.Unary("Relu", w), P + "attention.2.");
        w = Pointwise(b, w, P + "attention.3");
        // Softmax over time (the outermost, symbolic axis) per channel.
        var peak = b.Emit1("Reduce", [w], Attributes.From(("mode", "max"), ("axes", new long[] { 0 }), ("keepdims", true)));
        var exponent = b.Unary("Exp", b.Sub(w, peak));
        var total = b.Emit1("Reduce", [exponent], Attributes.From(("mode", "sum"), ("axes", new long[] { 0 }), ("keepdims", true)));
        w = b.Binary("Div", exponent, total);
        var sum = Attributes.From(("mode", "sum"), ("axes", new long[] { 0 }), ("keepdims", true));
        var mu = b.Emit1("Reduce", [b.Mul(x, w)], sum);
        var second = b.Emit1("Reduce", [b.Mul(b.Mul(x, x), w)], sum);
        var sigma = b.Unary("Sqrt", b.Emit1("Clip", [b.Sub(second, b.Mul(mu, mu))], Attributes.From(("min", 1e-5f))));
        b.Output(EmbeddingOutput, b.Linear(b.Concat(1, mu, sigma), P + "fc.weight", P + "fc.bias"));
        return b.Build();
    }

    // A kernel-1 Conv1d (weight [O, C, 1], bias [O]) over time-major rows.
    private static Value Pointwise(GraphBuilder b, Value x, string module)
        => b.Emit1("Conv1d", [x, b.Weight(module + ".weight"), b.Weight(module + ".bias")], Attributes.Empty);

    // Squeeze and excitation over [frames, C, frequency]: the channel means through two linear layers and a sigmoid scale the channels.
    private static Value SqueezeExcite(GraphBuilder b, Value x, string n)
    {
        long channels = x.Shape[1].Fixed;
        var pooled = b.Reshape(b.Emit1("Reduce", [x], Attributes.From(("mode", "mean"), ("axes", new long[] { 0, 2 }), ("keepdims", true))), 1, channels);
        var scale = b.Sigmoid(b.Linear(b.Unary("Relu", b.Linear(pooled, n + "0.weight", n + "0.bias")), n + "2.weight", n + "2.bias"));
        return b.Mul(x, b.Reshape(scale, 1, channels, 1));
    }

    // A PyTorch Conv2d over (frequency, time) on the time-major map [frames, C, frequency], strided equally in both.
    private static Value Conv2d(GraphBuilder b, Value x, string module, long stride, long padding = 1, bool bias = false)
    {
        var kernel = b.Permute(b.Weight(module + ".weight"), 0, 1, 3, 2);
        var attributes = Attributes.From(("strides", new[] { stride, stride }), ("pads", new[] { padding, padding, padding, padding }));
        Value[] inputs = bias ? [x, kernel, b.Weight(module + ".bias")] : [x, kernel];
        return b.Emit1("SequenceConv2d", inputs, attributes);
    }

    private static Value BatchNorm(GraphBuilder b, Value x, string n)
    {
        var inverse = b.Unary("Rsqrt", b.Scale(b.Weight(n + "running_var"), 1f, Eps));
        return b.Add(b.Mul(b.Mul(b.Sub(x, b.Weight(n + "running_mean")), inverse), b.Weight(n + "weight")), b.Weight(n + "bias"));
    }

    private static Value BatchNorm2d(GraphBuilder b, Value x, string n)
    {
        long channels = x.Shape[1].Fixed;
        Value Column(Value v) => b.Reshape(v, channels, 1);
        var inverse = b.Unary("Rsqrt", b.Scale(Column(b.Weight(n + "running_var")), 1f, Eps));
        var y = b.Mul(b.Sub(x, Column(b.Weight(n + "running_mean"))), inverse);
        return b.Add(b.Mul(y, Column(b.Weight(n + "weight"))), Column(b.Weight(n + "bias")));
    }
}
