using System.Runtime.InteropServices;
using Sharplight.Audio;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Emits the hybrid transformer Demucs network of one fixed-length segment as a graph of generic ops (see <see cref="SeparationGraphs"/>): per layer a spectral and a
/// waveform encoder (strided convolution, GELU, a residual dilated convolution branch, a 1 × 1 rewrite with a gated linear unit), a cross-domain transformer at the bottleneck
/// (self-attention and cross-attention layers over the spectral and waveform tokens with sinusoidal positions), and the mirrored decoders with transposed convolutions and
/// skip connections. Every convolution runs as a 2-D convolution in NCHW layout (the waveform as one row). Convolution weights of the importer are 4-D (<c>[O, C, KH, KW]</c>).
/// </summary>
internal static class DemucsGraph
{
    private const float NormEpsilon = 1e-5f;

    /// <summary>Builds the graph.</summary>
    /// <param name="p">The hyper-parameters.</param>
    /// <param name="weights">Finds a weight by name.</param>
    public static ComputeGraph Build(DemucsParameters p, Func<string, WeightInfo?> weights)
    {
        var b = new GraphBuilder(SeparationGraphs.SeparateGraph, weights);
        long frames = p.Frames;
        long bins = p.Bins;
        long length = p.SegmentSamples;
        int audio = p.AudioChannels;
        var x = b.Input(SeparationGraphs.SpectrumInput, InputRole.Media, DataType.F32, new Shape(1, 2 * audio, bins, frames));
        var xt = b.Reshape(b.Input(SeparationGraphs.WaveformInput, InputRole.Media, DataType.F32, new Shape(1, audio, length)), 1, audio, 1, length);
        var saved = new List<Value>();
        var savedTime = new List<Value>();
        var lengths = new List<long>();
        long freqs = bins;
        long current = length;
        for (int i = 0; i < p.Depth; i++)
        {
            using var _ = b.Group($"encoder.{i}");
            int channels = p.Channels * (int)Math.Pow(p.Growth, i);
            lengths.Add(current);
            long padded = current % p.Stride == 0 ? current : current + (p.Stride - (current % p.Stride));
            if (padded != current)
            {
                xt = b.Emit1("Pad", [xt], Attributes.From(("pads", new long[] { 0, 0, 0, 0, 0, 0, 0, padded - current }), ("mode", "constant"), ("value", 0f)));
            }

            xt = EncoderLayer(b, p, xt, $"tencoder.{i}.", channels, time: true, 0, padded / p.Stride);
            savedTime.Add(xt);
            current = padded / p.Stride;
            x = EncoderLayer(b, p, x, $"encoder.{i}.", channels, time: false, freqs / p.Stride, frames);
            freqs /= p.Stride;
            if (i == 0 && p.FrequencyEmbedding > 0)
            {
                // The learned frequency embedding, scaled as ScaledEmbedding and the model's freq_emb weight do.
                var embedding = b.Scale(b.Weight("freq_emb.embedding.weight"), p.EmbeddingScale * p.FrequencyEmbedding);
                x = b.Add(x, b.Reshape(b.Permute(embedding, 1, 0), 1, channels, freqs, 1));
            }

            saved.Add(x);
        }

        using (b.Group("transformer"))
        {
            int width = p.BottomChannels;
            int last = p.Channels * (int)Math.Pow(p.Growth, p.Depth - 1);
            long fr = freqs;
            var tokens = b.Reshape(x, 1, last, 1, fr * frames);
            tokens = Pointwise(b, tokens, "channel_upsampler");
            tokens = b.Reshape(b.Permute(b.Reshape(tokens, width, fr, frames), 2, 1, 0), frames * fr, width);
            var tokensTime = Pointwise(b, xt, "channel_upsampler_t");
            tokensTime = b.Permute(b.Reshape(tokensTime, width, current), 1, 0);
            tokens = TowerBlocks.LayerNorm(b, tokens, "crosstransformer.norm_in", NormEpsilon);
            tokensTime = TowerBlocks.LayerNorm(b, tokensTime, "crosstransformer.norm_in_t", NormEpsilon);
            tokens = b.Add(tokens, Constant(b, "position_2d", Position2D(width, (int)fr, (int)frames, p.MaxPeriod, p.PositionWeight), frames * fr, width));
            tokensTime = b.Add(tokensTime, Constant(b, "position_1d", Position1D((int)current, width, p.MaxPeriod, p.PositionWeight), current, width));
            for (int layer = 0; layer < p.TransformerLayers; layer++)
            {
                using var _ = b.Group($"crosstransformer.{layer}");
                bool cross = layer % 2 == 1;
                var before = tokens;
                tokens = TransformerLayer(b, p, tokens, cross ? tokensTime : null, $"crosstransformer.layers.{layer}.", cross);
                tokensTime = TransformerLayer(b, p, tokensTime, cross ? before : null, $"crosstransformer.layers_t.{layer}.", cross);
            }

            x = b.Reshape(b.Permute(b.Reshape(tokens, frames, fr, width), 2, 1, 0), 1, width, 1, fr * frames);
            x = b.Reshape(Pointwise(b, x, "channel_downsampler"), 1, last, fr, frames);
            xt = Pointwise(b, b.Reshape(b.Permute(tokensTime, 1, 0), 1, width, 1, current), "channel_downsampler_t");
        }

        for (int i = 0; i < p.Depth; i++)
        {
            using var _ = b.Group($"decoder.{i}");
            int channels = p.Channels * (int)Math.Pow(p.Growth, p.Depth - 1 - i);
            bool lastLayer = i == p.Depth - 1;
            x = DecoderLayer(b, p, x, saved[^1], $"decoder.{i}.", channels, time: false, freqs, frames, 0, lastLayer);
            saved.RemoveAt(saved.Count - 1);
            freqs *= p.Stride;
            long target = lengths[^1];
            lengths.RemoveAt(lengths.Count - 1);
            xt = DecoderLayer(b, p, xt, savedTime[^1], $"tdecoder.{i}.", channels, time: true, 0, 0, target, lastLayer);
            savedTime.RemoveAt(savedTime.Count - 1);
        }

        b.Output(SeparationGraphs.SpectrumOutput, x);
        b.Output(SeparationGraphs.WaveformOutput, b.Reshape(xt, p.Sources.Length * audio, length));
        return b.Build();
    }

    // The 2-D convolution: strides, pads (top, left, bottom, right) and dilations as the op's attributes.
    private static Value Conv(GraphBuilder b, Value x, string weight, long[] strides, long[] pads, long[]? dilations = null)
    {
        var attributes = Attributes.From(("strides", strides), ("pads", pads), ("dilations", dilations ?? [1, 1]));
        string bias = weight[..^"weight".Length] + "bias";
        return b.HasWeight(bias) ? b.Emit1("Conv", [x, b.Weight(weight), b.Weight(bias)], attributes) : b.Emit1("Conv", [x, b.Weight(weight)], attributes);
    }

    // A 1 × 1 convolution (a linear map over the channels) of weights name.weight / name.bias.
    private static Value Pointwise(GraphBuilder b, Value x, string name) => Conv(b, x, name + ".weight", [1, 1], [0, 0, 0, 0]);

    private static Value GroupNorm(GraphBuilder b, Value x, string name, int groups)
        => b.Emit1("GroupNorm", [x, b.Weight(name + ".weight"), b.Weight(name + ".bias")], Attributes.From(("groups", groups), ("eps", NormEpsilon)));

    // The gated linear unit over the channels (axis 1): the first half times the sigmoid of the second.
    private static Value Glu(GraphBuilder b, Value x, long channels)
        => b.Mul(b.Slice(x, 1, 0, channels), b.Sigmoid(b.Slice(x, 1, channels, channels)));

    // The residual branches: dilated convolution, group norm, GELU, 1 × 1 convolution, group norm, GLU, layer scale. x is [N, C, 1, T].
    private static Value Residual(GraphBuilder b, DemucsParameters p, Value x, string name, long channels)
    {
        for (int d = 0; d < p.DconvDepth; d++)
        {
            string n = $"{name}layers.{d}.";
            long dilation = 1L << d;
            var y = Conv(b, x, n + "0.weight", [1, 1], [0, dilation, 0, dilation], [1, dilation]);
            y = b.Unary("Gelu", GroupNorm(b, y, n + "1", 1));
            y = Glu(b, GroupNorm(b, Conv(b, y, n + "3.weight", [1, 1], [0, 0, 0, 0]), n + "4", 1), channels);
            x = b.Add(x, b.Mul(y, b.Reshape(b.Weight(n + "6.scale"), 1, channels, 1, 1)));
        }

        return x;
    }

    // The residual branches of a spectral tensor [1, C, F, T]: every frequency bin is its own batch row over time; of a waveform tensor [1, C, 1, T] directly.
    private static Value Branch(GraphBuilder b, DemucsParameters p, Value x, string name, long channels, bool time, long freqs, long frames)
    {
        if (time)
        {
            return Residual(b, p, x, name, channels);
        }

        var rows = b.Reshape(b.Permute(x, 0, 2, 1, 3), freqs, channels, 1, frames);
        rows = Residual(b, p, rows, name, channels);
        return b.Permute(b.Reshape(rows, 1, freqs, channels, frames), 0, 2, 1, 3);
    }

    private static Value EncoderLayer(GraphBuilder b, DemucsParameters p, Value x, string name, int channels, bool time, long freqs, long frames)
    {
        long pad = p.Kernel / 4;
        var y = time
            ? Conv(b, x, name + "conv.weight", [1, p.Stride], [0, pad, 0, pad])
            : Conv(b, x, name + "conv.weight", [p.Stride, 1], [pad, 0, pad, 0]);
        y = b.Unary("Gelu", y);
        y = Branch(b, p, y, name + "dconv.", channels, time, freqs, frames);
        return Glu(b, Pointwise(b, y, name + "rewrite"), channels);
    }

    private static Value DecoderLayer(GraphBuilder b, DemucsParameters p, Value x, Value skip, string name, int channels, bool time, long freqs, long frames, long target, bool last)
    {
        long pad = p.Kernel / 4;
        x = b.Add(x, skip);
        var y = time
            ? Conv(b, x, name + "rewrite.weight", [1, 1], [0, 1, 0, 1])
            : Conv(b, x, name + "rewrite.weight", [1, 1], [1, 1, 1, 1]);
        y = Glu(b, y, channels);
        y = Branch(b, p, y, name + "dconv.", channels, time, freqs, frames);
        Value z;
        if (time)
        {
            z = b.Emit1("ConvTranspose", [y, b.Weight(name + "conv_tr.weight"), b.Weight(name + "conv_tr.bias")], Attributes.From(("strides", new long[] { 1, p.Stride })));
            z = b.Slice(z, 3, pad, target);
        }
        else
        {
            z = b.Emit1("ConvTranspose", [y, b.Weight(name + "conv_tr.weight"), b.Weight(name + "conv_tr.bias")], Attributes.From(("strides", new long[] { p.Stride, 1 })));
            z = b.Slice(z, 2, pad, freqs * p.Stride);
        }

        return last ? z : b.Unary("Gelu", z);
    }

    // A pre-norm transformer layer over tokens [T, C]: self-attention, or cross-attention over the other stream's tokens, then the feed-forward, layer scales and a group norm.
    private static Value TransformerLayer(GraphBuilder b, DemucsParameters p, Value x, Value? other, string name, bool cross)
    {
        int width = p.BottomChannels;
        int heads = p.TransformerHeads;
        int headSize = width / heads;
        string attention = cross ? name + "cross_attn." : name + "self_attn.";
        var query = TowerBlocks.LayerNorm(b, x, name + "norm1", NormEpsilon);
        var source = cross ? TowerBlocks.LayerNorm(b, other!, name + "norm2", NormEpsilon) : query;
        Value Heads(Value input, string part) => b.Reshape(b.Linear(input, attention + part + ".weight", attention + part + ".bias"), input.Shape[0].Fixed, heads, headSize);
        long rows = x.Shape[0].Fixed;
        var context = b.Emit1("Attention", [Heads(query, "q"), Heads(source, "k"), Heads(source, "v")], Attributes.From(("scale", 1f / MathF.Sqrt(headSize)), ("causal", false)));
        var projected = b.Linear(b.Reshape(context, rows, width), attention + "out_proj.weight", attention + "out_proj.bias");
        x = b.Add(x, b.Mul(projected, b.Weight(name + "gamma_1.scale")));
        var h = TowerBlocks.LayerNorm(b, x, name + (cross ? "norm3" : "norm2"), NormEpsilon);
        h = b.Linear(b.Unary("Gelu", b.Linear(h, name + "linear1.weight", name + "linear1.bias")), name + "linear2.weight", name + "linear2.bias");
        x = b.Add(x, b.Mul(h, b.Weight(name + "gamma_2.scale")));
        // The group norm over all tokens and channels together (one group).
        var grouped = b.Reshape(b.Permute(x, 1, 0), 1, width, rows);
        grouped = GroupNorm(b, grouped, name + "norm_out", 1);
        return b.Permute(b.Reshape(grouped, width, rows), 1, 0);
    }

    private static Value Constant(GraphBuilder b, string name, float[] values, long rows, long columns)
        => b.Constant(name, DataType.F32, new Shape(rows, columns), MemoryMarshal.AsBytes(values.AsSpan()).ToArray());

    // create_sin_embedding: [length, dim] with the cosines of the phases in the first half of the channels and the sines in the second.
    internal static float[] Position1D(int length, int dim, double maxPeriod, float weight)
    {
        int half = dim / 2;
        var result = new float[length * dim];
        for (int t = 0; t < length; t++)
        {
            for (int k = 0; k < half; k++)
            {
                double phase = t / Math.Pow(maxPeriod, (double)k / (half - 1));
                result[(t * dim) + k] = (float)(Math.Cos(phase) * weight);
                result[(t * dim) + half + k] = (float)(Math.Sin(phase) * weight);
            }
        }

        return result;
    }

    // create_2d_sin_embedding for tokens in (time, frequency) order: [width · height, dim], the first half of the channels encodes the time position (sine, cosine pairs),
    // the second half the frequency position.
    internal static float[] Position2D(int dim, int height, int width, double maxPeriod, float weight)
    {
        int half = dim / 2;
        var result = new float[height * width * dim];
        var division = new double[half / 2];
        for (int k = 0; k < division.Length; k++)
        {
            division[k] = Math.Exp(2 * k * -(Math.Log(maxPeriod) / half));
        }

        for (int t = 0; t < width; t++)
        {
            for (int f = 0; f < height; f++)
            {
                int row = ((t * height) + f) * dim;
                for (int k = 0; k < division.Length; k++)
                {
                    result[row + (2 * k)] = (float)(Math.Sin(t * division[k]) * weight);
                    result[row + (2 * k) + 1] = (float)(Math.Cos(t * division[k]) * weight);
                    result[row + half + (2 * k)] = (float)(Math.Sin(f * division[k]) * weight);
                    result[row + half + (2 * k) + 1] = (float)(Math.Cos(f * division[k]) * weight);
                }
            }
        }

        return result;
    }
}
