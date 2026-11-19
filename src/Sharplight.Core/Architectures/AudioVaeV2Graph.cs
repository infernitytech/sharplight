using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Emits VoxCPM2's AudioVAE V2 as streaming graphs of generic ops over time-major signals: every causal convolution takes its left context (the zero padding of a
/// whole-signal run) from a cache input <c>{prefix}cache.{n}</c> and hands the context of the next run to the output <c>{prefix}cache_out.{n}</c> (the naming of
/// <see cref="VibeVoiceTokenizerGraph"/>, so <see cref="VibeVoiceConvCaches"/> carries them), so a stream of fixed-size runs equals one run over the whole signal.
/// The encoder (a depthwise-separable causal convolution stack with snake activations, strides <see cref="AudioVaeV2Parameters.EncoderRates"/>) turns input-rate audio
/// into the mean of the latent distribution; the decoder (depthwise first convolution, transposed convolutions with dilated depthwise residual units, each block's
/// input scaled and shifted by the rows of the output rate's bucket in the sample-rate conditioning tables, tanh at the end) turns latents into output-rate audio.
/// Weights under <c>audio_vae.</c>, named as the checkpoint names them (weight normalizations folded, snake parameters as <c>[C]</c>).
/// </summary>
internal static class AudioVaeV2Graph
{
    /// <summary>The prefix of the AudioVAE's weights.</summary>
    public const string WeightPrefix = "audio_vae.";

    /// <summary>The prefix of the decoder graph's caches.</summary>
    public const string DecoderCachePrefix = "vae_decoder.";

    /// <summary>The prefix of the encoder graph's caches.</summary>
    public const string EncoderCachePrefix = "vae_encoder.";

    /// <summary>The audio input of the encoder graph.</summary>
    public const string AudioInput = "audio";

    /// <summary>The latents input of the decoder graph and output of the encoder graph.</summary>
    public const string LatentsName = "latents";

    /// <summary>The audio output of the decoder graph.</summary>
    public const string AudioOutput = "audio";

    private const float SnakeEpsilon = 1e-9f;
    private const int Kernel = 7;

    /// <summary>Builds the decoder graph: <paramref name="frames"/> latent frames [frames, latent] to audio [frames · decoder hop, 1].</summary>
    /// <param name="name">The graph name.</param>
    /// <param name="p">The sizes.</param>
    /// <param name="weights">Finds a weight by name.</param>
    /// <param name="frames">The latent frames per run.</param>
    public static ComputeGraph BuildDecoder(string name, AudioVaeV2Parameters p, Func<string, WeightInfo?> weights, int frames)
    {
        var b = new GraphBuilder(name, weights);
        var emit = new Emitter(b, DecoderCachePrefix);
        const string W = WeightPrefix + "decoder.";
        var x = b.Input(LatentsName, InputRole.Other, DataType.F32, new Shape(frames, p.Latent));
        long rows = frames;
        x = emit.Conv(x, W + "model.0", p.Latent, rows, Kernel, 1, 1, p.Latent);
        x = emit.Conv(x, W + "model.1", p.Latent, rows, 1, 1, 1, 1);
        long channels = p.DecoderDim;
        for (int i = 0; i < p.DecoderRates.Length; i++)
        {
            int layer = 2 + i;
            using var _ = b.Group($"vae.decoder.block.{i}");
            if (p.SampleRateBucket >= 0)
            {
                string c = $"{W}sr_cond_model.{layer}.";
                x = b.Add(b.Mul(x, b.Slice(b.Weight(c + "scale_embed.weight"), 0, p.SampleRateBucket, 1)), b.Slice(b.Weight(c + "bias_embed.weight"), 0, p.SampleRateBucket, 1));
            }

            string n = $"{W}model.{layer}.block.";
            long stride = p.DecoderRates[i];
            long output = channels / 2;
            x = Snake(b, x, n + "0.alpha");
            x = emit.ConvTranspose(x, n + "1", channels, rows, stride);
            rows *= stride;
            channels = output;
            x = ResidualUnits(b, emit, x, n, 2, channels, rows);
        }

        int last = 2 + p.DecoderRates.Length;
        x = Snake(b, x, $"{W}model.{last}.alpha");
        x = emit.Conv(x, $"{W}model.{last + 1}", channels, rows, Kernel, 1, 1, 1);
        b.Output(AudioOutput, b.Unary("Tanh", x));
        return b.Build();
    }

    /// <summary>Builds the encoder graph: <paramref name="samples"/> input-rate samples [samples, 1] (a multiple of the encoder hop) to latent means [samples / hop, latent].</summary>
    /// <param name="name">The graph name.</param>
    /// <param name="p">The sizes.</param>
    /// <param name="weights">Finds a weight by name.</param>
    /// <param name="samples">The samples per run.</param>
    public static ComputeGraph BuildEncoder(string name, AudioVaeV2Parameters p, Func<string, WeightInfo?> weights, int samples)
    {
        var b = new GraphBuilder(name, weights);
        var emit = new Emitter(b, EncoderCachePrefix);
        const string W = WeightPrefix + "encoder.";
        var x = b.Input(AudioInput, InputRole.Media, DataType.F32, new Shape(samples, 1));
        long rows = samples;
        x = emit.Conv(x, W + "block.0", 1, rows, Kernel, 1, 1, 1);
        long channels = p.EncoderDim;
        for (int i = 0; i < p.EncoderRates.Length; i++)
        {
            using var _ = b.Group($"vae.encoder.block.{i}");
            string n = $"{W}block.{i + 1}.block.";
            long stride = p.EncoderRates[i];
            x = ResidualUnits(b, emit, x, n, 0, channels, rows);
            x = Snake(b, x, n + "3.alpha");
            x = emit.Conv(x, n + "4", channels, rows, (int)(2 * stride), stride, 1, 1);
            rows /= stride;
            channels *= 2;
        }

        b.Output(LatentsName, emit.Conv(x, W + "fc_mu", channels, rows, 3, 1, 1, 1));
        return b.Build();
    }

    // Three residual units (dilations 1, 3, 9) at {prefix}{first}, {first + 1}, {first + 2}: x + conv1x1(snake(conv7_dilated_depthwise(snake(x)))).
    private static Value ResidualUnits(GraphBuilder b, Emitter emit, Value x, string prefix, int first, long channels, long rows)
    {
        long[] dilations = [1, 3, 9];
        for (int u = 0; u < dilations.Length; u++)
        {
            string r = $"{prefix}{first + u}.block.";
            var y = emit.Conv(Snake(b, x, r + "0.alpha"), r + "1", channels, rows, Kernel, 1, dilations[u], channels);
            y = emit.Conv(Snake(b, y, r + "2.alpha"), r + "3", channels, rows, 1, 1, 1, 1);
            x = b.Add(x, y);
        }

        return x;
    }

    private static Value Snake(GraphBuilder b, Value x, string alpha) => SnakeActivation.Apply(b, x, b.Weight(alpha), SnakeEpsilon);

    /// <summary>Emits the causal convolutions with their caches.</summary>
    private sealed class Emitter
    {
        private readonly GraphBuilder _b;
        private readonly string _prefix;
        private int _caches;

        public Emitter(GraphBuilder b, string prefix)
        {
            _b = b;
            _prefix = prefix;
        }

        // The causal convolution {name}.weight / {name}.bias whose left context ((kernel - 1) · dilation + 1 - stride rows of the input) comes from a cache input.
        public Value Conv(Value x, string name, long channels, long rows, int kernel, long stride, long dilation, long groups)
        {
            long pad = ((kernel - 1) * dilation) + 1 - stride;
            var input = x;
            if (pad > 0)
            {
                input = _b.Concat(0, Cache(channels, pad), x);
                Retain(input, rows, pad);
            }

            return _b.Emit1("Conv1d", [input, _b.Weight(name + ".weight"), _b.Weight(name + ".bias")],
                Attributes.From(("stride", stride), ("dilation", dilation), ("pad_left", 0L), ("pad_right", 0L), ("groups", groups)));
        }

        // The causal transposed convolution (kernel = 2 · stride, the last stride outputs dropped): the previous input row supplies the left context; the output rows of
        // the new inputs are kept.
        public Value ConvTranspose(Value x, string name, long channels, long rows, long stride)
        {
            var input = _b.Concat(0, Cache(channels, 1), x);
            Retain(input, rows, 1);
            var y = _b.Emit1("ConvTranspose1d", [input, _b.Weight(name + ".weight"), _b.Weight(name + ".bias")],
                Attributes.From(("stride", stride), ("crop_left", 0L), ("crop_right", stride)));
            return _b.Slice(y, 0, stride, rows * stride);
        }

        private Value Cache(long channels, long rows)
            => _b.Input($"{_prefix}{VibeVoiceTokenizerGraph.CacheInfix}{_caches}", InputRole.Other, DataType.F32, new Shape(rows, channels));

        // The last pad rows of the padded input are the next run's context.
        private void Retain(Value padded, long rows, long pad)
            => _b.Output($"{_prefix}{VibeVoiceTokenizerGraph.CacheOutInfix}{_caches++}", _b.Slice(padded, 0, rows, pad));
    }
}
