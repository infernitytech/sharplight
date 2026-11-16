using Sharplight.Audio;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Emits the WavTokenizer decoder as a graph of generic ops over a time-major signal with a symbolic number of frames <c>T</c>: codebook
/// lookup, input convolution, the PosNet stack (residual blocks of group norm, SiLU and convolution; one single-head attention block over the
/// frames), a layer norm, the ConvNeXt blocks (depthwise convolution, layer norm, GELU feed-forward, layer scale, residual), the head, and the
/// frequency-domain half of the inverse STFT: <c>exp(log-magnitude)</c> (clipped) and the phase become a one-sided spectrum, an inverse real DFT
/// per frame gives the windowed frames. The windowed frames (<see cref="FramesOutput"/>, <c>[T, n_fft]</c>) are overlap-added and
/// normalized on the host (<c>Sharplight.Audio.OverlapAdd</c>).
/// </summary>
internal static class WavTokenizerGraph
{
    /// <summary>The graph name.</summary>
    public const string GraphName = "codec_decode";

    /// <summary>The graph input: the codec tokens, i32 <c>[T]</c>.</summary>
    public const string CodesInput = "codes";

    /// <summary>The graph output: the windowed inverse-DFT frames, f32 <c>[T, n_fft]</c>.</summary>
    public const string FramesOutput = "frames";

    /// <summary>The symbol of the number of frames.</summary>
    public const string FrameSymbol = "T";

    /// <summary>Builds the graph reading the codec's weights as <c>{prefix}name</c>.</summary>
    /// <param name="p">The hyper-parameters.</param>
    /// <param name="weights">Finds a weight by its full name.</param>
    /// <param name="prefix">The prefix of the codec's weights in the model.</param>
    public static ComputeGraph Build(WavTokenizerParameters p, Func<string, WeightInfo?> weights, string prefix)
    {
        var b = new GraphBuilder(GraphName, weights);
        Value W(string name) => b.Weight(prefix + name);
        int c = p.HiddenSize;

        var codes = b.Input(CodesInput, InputRole.TokenIds, DataType.I32, new Shape(Dim.Sym(FrameSymbol)));
        var x = b.Gather(W("embedding.weight"), codes, "codes_embedding");
        x = Conv(b, x, W("input_conv.weight"), W("input_conv.bias"));

        for (int il = 0; il < p.PosnetLayers; il++)
        {
            string n = $"posnet.{il}.";
            switch (il)
            {
                case 2:
                    x = Attention(b, p, x, n, prefix);
                    break;
                case 5:
                    x = GroupNorm(b, p, x, W(n + "norm.weight"), W(n + "norm.bias"));
                    break;
                default:
                    var h = Conv(b, b.Silu(GroupNorm(b, p, x, W(n + "norm1.weight"), W(n + "norm1.bias"))), W(n + "conv1.weight"), W(n + "conv1.bias"));
                    h = Conv(b, b.Silu(GroupNorm(b, p, h, W(n + "norm2.weight"), W(n + "norm2.bias"))), W(n + "conv2.weight"), W(n + "conv2.bias"));
                    x = b.Add(x, h);
                    break;
            }
        }

        x = LayerNorm(b, x, W("embedding_norm.weight"), W("embedding_norm.bias"), p.NormEps);
        for (int il = 0; il < p.Blocks; il++)
        {
            string n = $"blocks.{il}.";
            var h = Conv(b, x, W(n + "dw.weight"), W(n + "dw.bias"), groups: c);
            h = LayerNorm(b, h, W(n + "norm.weight"), W(n + "norm.bias"), p.NormEps);
            h = b.Linear(h, prefix + n + "pw1.weight", prefix + n + "pw1.bias");
            h = b.Unary("GeluTanh", h);
            h = b.Linear(h, prefix + n + "pw2.weight", prefix + n + "pw2.bias");
            x = b.Add(x, b.Mul(h, W(n + "gamma")));
        }

        x = LayerNorm(b, x, W("output_norm.weight"), W("output_norm.bias"), p.NormEps);
        var head = b.Linear(x, prefix + "head.weight", prefix + "head.bias");                                      // [T, 2 * bins]

        int bins = p.Bins;
        var magnitude = b.Emit1("Clip", [b.Unary("Exp", b.Slice(head, 1, 0, bins))], Attributes.From(("max", 100f)));
        var phase = b.Slice(head, 1, bins, bins);
        var real = b.Reshape(b.Mul(magnitude, b.Unary("Cos", phase)), 0, bins, 1);
        var imag = b.Reshape(b.Mul(magnitude, b.Unary("Sin", phase)), 0, bins, 1);
        var spectrum = b.Concat(2, real, imag);                                                                     // [T, bins, 2]
        var frames = b.Emit1("Dft", [spectrum], Attributes.From(("length", (long)p.NFft), ("out_bins", (long)p.NFft), ("inverse", true), ("real_output", true)));
        frames = b.Reshape(frames, 0, p.NFft);
        var window = b.Constant("hann_window", DataType.F32, new Shape(p.NFft), Floats(Window.Hann(p.NFft)));
        b.Output(FramesOutput, b.Mul(frames, window));
        return b.Build();
    }

    /// <summary>A 1-D convolution with 'same' padding of a kernel of odd length.</summary>
    private static Value Conv(GraphBuilder b, Value x, Value weight, Value bias, int groups = 1)
    {
        long taps = weight.Shape[2].Fixed;
        var attributes = Attributes.From(("pad_left", (taps - 1) / 2), ("pad_right", (taps - 1) / 2), ("groups", (long)groups));
        return b.Emit1("Conv1d", [x, weight, bias], attributes);
    }

    private static Value LayerNorm(GraphBuilder b, Value x, Value weight, Value bias, float eps)
        => b.Emit1("LayerNorm", [x, weight, bias], Attributes.From(("eps", eps)));

    /// <summary>
    /// Group normalization of a time-major signal <c>[T, C]</c>: statistics over all frames and the channels of a group, then the per-channel
    /// scale and bias (reductions over the symbolic frame axis).
    /// </summary>
    private static Value GroupNorm(GraphBuilder b, WavTokenizerParameters p, Value x, Value weight, Value bias)
    {
        int groups = p.GroupNormGroups;
        long channels = x.Shape[1].Fixed;
        var grouped = b.Reshape(x, 0, groups, channels / groups);
        var axes = Attributes.From(("mode", "mean"), ("axes", new long[] { 0, 2 }), ("keepdims", true));
        var centered = b.Sub(grouped, b.Emit1("Reduce", [grouped], axes));
        var variance = b.Emit1("Reduce", [b.Unary("Square", centered)], axes);
        var normalized = b.Mul(centered, b.Unary("Rsqrt", b.Scale(variance, 1f, p.GroupNormEps)));
        return b.Add(b.Mul(b.Reshape(normalized, 0, channels), weight), bias);
    }

    /// <summary>The attention block: group norm, 1x1 convolutions to q, k and v, one head over all frames, output projection, residual.</summary>
    private static Value Attention(GraphBuilder b, WavTokenizerParameters p, Value x, string n, string prefix)
    {
        var h = GroupNorm(b, p, x, b.Weight(prefix + n + "attn_norm.weight"), b.Weight(prefix + n + "attn_norm.bias"));
        long c = p.HiddenSize;
        Value Project(string name) => b.Reshape(b.Linear(h, prefix + n + name + ".weight", prefix + n + name + ".bias"), 0, 1, c);
        var attended = b.Emit1("Attention", [Project("q"), Project("k"), Project("v")], Attributes.From(("scale", 1f / MathF.Sqrt(c)), ("causal", false)));
        var o = b.Linear(b.Reshape(attended, 0, c), prefix + n + "o.weight", prefix + n + "o.bias");
        return b.Add(x, o);
    }

    private static byte[] Floats(float[] values)
    {
        var bytes = new byte[values.Length * sizeof(float)];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return bytes;
    }
}
