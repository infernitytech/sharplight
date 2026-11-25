using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Emits the VibeVoice speech tokenizers as graphs of generic ops over time-major signals: the encoder (stem convolution, ConvNeXt-style stages joined by strided convolutions, head
/// convolution) maps 24 kHz audio [frames · hop, 1] to latents [frames, latent]; the decoder (stem, stages joined by transposed convolutions, head) maps latents back to audio.
/// Both are causal, and a graph covers a fixed number of frames: every causal convolution reads its left context from an input <c>{prefix}cache.{n}</c> and writes the context for the next
/// run to the output <c>{prefix}cache_out.{n}</c> (<see cref="VibeVoiceConvCaches"/> carries them from run to run), so a stream of runs equals one run over the whole signal
/// (transformers' padding cache), and a run over a whole signal needs only zero caches.
/// </summary>
internal static class VibeVoiceTokenizerGraph
{
    /// <summary>The infix of the cache inputs, between the prefix and the index.</summary>
    public const string CacheInfix = "cache.";

    /// <summary>The infix of the cache outputs.</summary>
    public const string CacheOutInfix = "cache_out.";

    /// <summary>Emits the encoder of the tokenizer whose weights are named <c>{weights}encoder.*</c>.</summary>
    /// <param name="b">The graph builder.</param>
    /// <param name="p">The tokenizer's hyper-parameters.</param>
    /// <param name="audio">The audio [frames · hop, 1].</param>
    /// <param name="weights">The weight prefix, for example <c>acoustic.</c>.</param>
    /// <param name="prefix">The prefix of this tokenizer's cache inputs and outputs in the graph.</param>
    /// <returns>The latents [frames, latent].</returns>
    public static Value Encode(GraphBuilder b, VibeVoiceTokenizerParameters p, Value audio, string weights, string prefix)
    {
        var emit = new Emitter(b, p, weights + "encoder.", prefix);
        long rows = audio.Shape[0].Fixed;
        var ratios = p.Ratios.Reverse().ToArray();
        var x = emit.Conv(audio, "down.0", 1, rows, p.Kernel, 1, 1);
        x = emit.Stage(x, 0, p.Depths[0], p.Filters, rows);
        for (int s = 1; s < p.Stages; s++)
        {
            long ratio = ratios[s - 1];
            long channels = (long)p.Filters << (s - 1);
            x = emit.Conv(x, $"down.{s}", channels, rows, (int)(2 * ratio), ratio, 1);
            rows /= ratio;
            x = emit.Stage(x, s, p.Depths[s], channels * 2, rows);
        }

        return emit.Conv(x, "head", (long)p.Filters << (p.Stages - 1), rows, p.Kernel, 1, 1);
    }

    /// <summary>Emits the decoder of the tokenizer whose weights are named <c>{weights}decoder.*</c>.</summary>
    /// <param name="b">The graph builder.</param>
    /// <param name="p">The tokenizer's hyper-parameters.</param>
    /// <param name="latents">The latents [frames, latent].</param>
    /// <param name="weights">The weight prefix, for example <c>acoustic.</c>.</param>
    /// <param name="prefix">The prefix of this tokenizer's cache inputs and outputs in the graph.</param>
    /// <returns>The audio [frames · hop, 1].</returns>
    public static Value Decode(GraphBuilder b, VibeVoiceTokenizerParameters p, Value latents, string weights, string prefix)
    {
        var emit = new Emitter(b, p, weights + "decoder.", prefix);
        long rows = latents.Shape[0].Fixed;
        var depths = p.Depths.Reverse().ToArray();
        long top = (long)p.Filters << (p.Stages - 1);
        var x = emit.Conv(latents, "up.0", p.Latent, rows, p.Kernel, 1, 1);
        x = emit.Stage(x, 0, depths[0], top, rows);
        for (int s = 1; s < p.Stages; s++)
        {
            long ratio = p.Ratios[s - 1];
            x = emit.ConvTranspose(x, $"up.{s}", top >> (s - 1), rows, ratio);
            rows *= ratio;
            x = emit.Stage(x, s, depths[s], top >> s, rows);
        }

        return emit.Conv(x, "head", p.Filters, rows, p.Kernel, 1, 1);
    }

    /// <summary>The ConvNeXt-style block count and channel layout of an encoder stage are the checkpoint's; this emits them.</summary>
    private sealed class Emitter
    {
        private readonly GraphBuilder _b;
        private readonly VibeVoiceTokenizerParameters _p;
        private readonly string _weights;
        private readonly string _prefix;
        private int _caches;

        public Emitter(GraphBuilder b, VibeVoiceTokenizerParameters p, string weights, string prefix)
        {
            _b = b;
            _p = p;
            _weights = weights;
            _prefix = prefix;
        }

        // A stage: the blocks that run at one rate.
        public Value Stage(Value x, int stage, int blocks, long channels, long rows)
        {
            for (int j = 0; j < blocks; j++)
            {
                x = Block(x, $"{_weights}stage.{stage}.{j}.", channels, rows);
            }

            return x;
        }

        // x + gamma · depthwise_conv(norm(x)), then x + ffn_gamma · ffn(ffn_norm(x)).
        private Value Block(Value x, string w, long channels, long rows)
        {
            using var _ = _b.Group(w.TrimEnd('.'));
            var h = _b.RmsNorm(x, _b.Weight(w + "norm.weight"), _p.Eps);
            h = Conv(h, w + "mixer", channels, rows, _p.Kernel, 1, channels, absolute: true);
            x = _b.Add(x, _b.Mul(h, _b.Weight(w + "gamma")));
            h = _b.RmsNorm(x, _b.Weight(w + "ffn_norm.weight"), _p.Eps);
            h = _b.Linear(h, w + "ffn1.weight", w + "ffn1.bias");
            h = _b.Linear(_b.Unary("Gelu", h), w + "ffn2.weight", w + "ffn2.bias");
            return _b.Add(x, _b.Mul(h, _b.Weight(w + "ffn_gamma")));
        }

        // The causal convolution (weights <c>name.weight</c>, <c>name.bias</c>) whose left context (kernel - stride rows of the input) comes from a cache input.
        public Value Conv(Value x, string name, long channels, long rows, int kernel, long stride, long groups, bool absolute = false)
        {
            string weights = absolute ? name : _weights + name;
            long pad = kernel - stride;
            var input = x;
            if (pad > 0)
            {
                input = _b.Concat(0, Cache(channels, pad), x);
                Retain(input, rows, pad);
            }

            return _b.Emit1("Conv1d", [input, _b.Weight(weights + ".weight"), _b.Weight(weights + ".bias")],
                Attributes.From(("stride", stride), ("dilation", 1L), ("pad_left", 0L), ("pad_right", 0L), ("groups", groups)));
        }

        // The causal transposed convolution (kernel = 2 · stride): the previous input row supplies the left context; the output rows of the new inputs are kept.
        public Value ConvTranspose(Value x, string name, long channels, long rows, long stride)
        {
            string weights = _weights + name;
            var input = _b.Concat(0, Cache(channels, 1), x);
            Retain(input, rows, 1);
            var y = _b.Emit1("ConvTranspose1d", [input, _b.Weight(weights + ".weight"), _b.Weight(weights + ".bias")],
                Attributes.From(("stride", stride), ("crop_left", 0L), ("crop_right", stride)));
            return _b.Slice(y, 0, stride, rows * stride);
        }

        private Value Cache(long channels, long rows)
            => _b.Input($"{_prefix}{CacheInfix}{_caches}", InputRole.Other, DataType.F32, new Shape(rows, channels));

        // The last <paramref name="pad"/> rows of the padded input are the next run's context.
        private void Retain(Value padded, long rows, long pad)
            => _b.Output($"{_prefix}{CacheOutInfix}{_caches++}", _b.Slice(padded, 0, rows, pad));
    }
}
