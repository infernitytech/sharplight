using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;
using Sharplight.VideoGeneration;

namespace Sharplight.Architectures;

/// <summary>
/// Builds the decoder of an LTX-Video VAE (<c>AutoencoderKLLTXVideo</c> with a non-causal decoder) as one graph over the whole latent
/// video (<see cref="VideoDiffusionGraphs.VaeDecode"/>), on frames-as-batch activations [T, C, H, W]: 3-D convolutions with the first and
/// last frame replicated in time and zero spatial padding (the kernel's time taps folded into its input channels, as for Wan),
/// channel RMS norms, residual blocks with a layer-normed 1×1×1 shortcut where the width changes, depth-to-space 2×2×2 upsamplers
/// that drop the first new frame, and the 4×4 pixel unpatching; the latent statistics undo the pipeline's normalization.
/// Timestep-conditioned decoders (LTX-Video 0.9.5 and later) embed the decode timestep (× <c>timestep_scale_multiplier</c>) per block
/// and modulate every residual block's two norms and the output norm with it; their upsamplers may add the depth-to-space shuffled
/// input as a residual and halve the channels (<c>upsample_residual</c>, <c>upsample_factor</c>).
/// </summary>
/// <remarks>
/// Canonical weights (diffusers names under <c>vae.</c>): <c>latents_mean</c> and <c>latents_std</c> f32 [C]; convolutions
/// <c>{name}.conv.weight</c> [O, kt·C, kh, kw] with <c>{name}.conv.bias</c>; <c>{name}.norm3.weight|bias</c> of the shortcut layer norms;
/// timestep conditioning: <c>decoder.timestep_scale_multiplier</c> [1], <c>{block}.time_embedder.timestep_embedder.linear_1|2</c>,
/// <c>{resnet}.scale_shift_table</c> f32 [4, C], <c>decoder.scale_shift_table</c> f32 [2, C].
/// </remarks>
internal sealed class LtxVaeGraphBuilder
{
    private const string P = "vae.";
    private const float RmsEps = 1e-8f;
    private const float LayerNormEps = 1e-6f;

    private readonly GraphBuilder _b;
    private readonly int _latentChannels;
    private readonly int _patchSize;
    private readonly float _decodeTimestep;
    private readonly bool[] _upsampleResidual;
    private readonly int[] _upsampleFactor;
    private Value? _timestep;

    private LtxVaeGraphBuilder(Func<string, WeightInfo?> weights, int latentChannels, int patchSize, float decodeTimestep, bool[] upsampleResidual, int[] upsampleFactor)
    {
        _b = new GraphBuilder(VideoDiffusionGraphs.VaeDecode, weights);
        _latentChannels = latentChannels;
        _patchSize = patchSize;
        _decodeTimestep = decodeTimestep;
        _upsampleResidual = upsampleResidual;
        _upsampleFactor = upsampleFactor;
    }

    /// <summary>Builds the whole-video decoder graph.</summary>
    /// <param name="weights">Canonical weight lookup.</param>
    /// <param name="latentChannels">Latent channels.</param>
    /// <param name="patchSize">Pixel patch size of the output (4).</param>
    /// <param name="latentFrames">Latent frames.</param>
    /// <param name="latentHeight">Latent height.</param>
    /// <param name="latentWidth">Latent width.</param>
    /// <param name="decodeTimestep">The decode timestep of a timestep-conditioned decoder (diffusers' <c>decode_timestep</c>, default 0).</param>
    /// <param name="upsampleResidual">Per up block (decoder order): whether its upsampler adds the shuffled input; empty for none.</param>
    /// <param name="upsampleFactor">Per up block (decoder order): the channel reduction of its upsampler; empty for 1.</param>
    public static ComputeGraph BuildDecoder(Func<string, WeightInfo?> weights, int latentChannels, int patchSize, long latentFrames, long latentHeight, long latentWidth,
        float decodeTimestep = 0f, bool[]? upsampleResidual = null, int[]? upsampleFactor = null)
        => new LtxVaeGraphBuilder(weights, latentChannels, patchSize, decodeTimestep, upsampleResidual ?? [], upsampleFactor ?? []).Decoder(latentFrames, latentHeight, latentWidth);

    private ComputeGraph Decoder(long frames, long height, long width)
    {
        int c = _latentChannels;
        var latent = _b.Input(VideoDiffusionGraphs.Latent, InputRole.Other, DataType.F32, new Shape(c, frames, height, width));
        var x = _b.Contiguous(_b.Permute(latent, 1, 0, 2, 3));
        x = _b.Add(_b.Mul(x, ChannelWeight(P + "latents_std", c)), ChannelWeight(P + "latents_mean", c));
        x = Conv(x, P + "decoder.conv_in");
        if (_b.HasWeight(P + "decoder.timestep_scale_multiplier"))
        {
            // diffusers: temb = decode_timestep · multiplier, embedded per block.
            var value = _b.Constant("decode_timestep", DataType.F32, new Shape(1), BitConverter.GetBytes(_decodeTimestep));
            _timestep = _b.Mul(value, _b.Reshape(_b.Weight(P + "decoder.timestep_scale_multiplier"), 1));
        }

        var midTime = TimeEmbedding(P + "decoder.mid_block.time_embedder");
        for (int r = 0; _b.HasWeight($"{P}decoder.mid_block.resnets.{r}.conv1.conv.weight"); r++)
        {
            x = ResidualBlock(x, $"{P}decoder.mid_block.resnets.{r}", midTime);
        }

        for (int i = 0; _b.HasWeight($"{P}decoder.up_blocks.{i}.resnets.0.conv1.conv.weight"); i++)
        {
            using var _ = _b.Group($"up.{i}");
            string p = $"{P}decoder.up_blocks.{i}";
            if (_b.HasWeight(p + ".conv_in.conv1.conv.weight"))
            {
                if (_b.HasWeight(p + ".conv_in.scale_shift_table"))
                {
                    throw new NotSupportedException("LTX-Video VAE: timestep-conditioned up-block input layers are not supported.");
                }

                x = ResidualBlock(x, p + ".conv_in");
            }

            var time = TimeEmbedding(p + ".time_embedder");
            if (_b.HasWeight(p + ".upsamplers.0.conv.conv.weight"))
            {
                x = Upsample(x, p + ".upsamplers.0", i < _upsampleResidual.Length && _upsampleResidual[i], i < _upsampleFactor.Length ? _upsampleFactor[i] : 1);
            }

            for (int r = 0; _b.HasWeight($"{p}.resnets.{r}.conv1.conv.weight"); r++)
            {
                x = ResidualBlock(x, $"{p}.resnets.{r}", time);
            }
        }

        x = RmsNorm(x);
        if (TimeEmbedding(P + "decoder.time_embedder") is { } final)
        {
            // shift, scale = table + temb.
            long channels = x.Shape[1].Fixed;
            var table = _b.Add(_b.Reshape(_b.Weight(P + "decoder.scale_shift_table"), 2, channels), _b.Reshape(final, 2, channels));
            var shift = _b.Reshape(_b.Slice(table, 0, 0, 1), 1, channels, 1, 1);
            var scale = _b.Reshape(_b.Slice(table, 0, 1, 1), 1, channels, 1, 1);
            x = _b.Add(_b.Mul(x, _b.Scale(scale, 1f, 1f)), shift);
        }

        x = Conv(_b.Silu(x), P + "decoder.conv_out");
        x = Unpatchify(x);
        var clamped = _b.Emit1("Clip", [x], Attributes.From(("min", -1f), ("max", 1f)));
        _b.Output(VideoDiffusionGraphs.Frames, _b.Scale(clamped, 0.5f, 0.5f));
        return _b.Build();
    }

    /// <summary>A non-causal 3-D convolution over all frames: the first and last frame repeated (kt − 1) / 2 times, zero spatial padding.</summary>
    private Value Conv(Value x, string prefix)
    {
        var weight = _b.Weight(prefix + ".conv.weight");
        long channels = x.Shape[1].Fixed;
        long taps = weight.Shape[1].Fixed / channels;
        long half = weight.Shape[2].Fixed / 2;
        var attributes = Attributes.From(("strides", new[] { 1L, 1L }), ("pads", new[] { half, half, half, half }));
        var bias = _b.TryWeight(prefix + ".conv.bias");
        var input = x;
        if (taps > 1)
        {
            long frames = x.Shape[0].Fixed;
            long side = (taps - 1) / 2;
            var first = _b.Slice(x, 0, 0, 1);
            var last = _b.Slice(x, 0, frames - 1, 1);
            var parts = new List<Value>();
            parts.AddRange(Enumerable.Repeat(first, (int)side));
            parts.Add(x);
            parts.AddRange(Enumerable.Repeat(last, (int)side));
            var padded = _b.Concat(0, [.. parts]);
            var windows = new Value[taps];
            for (int j = 0; j < taps; j++)
            {
                windows[j] = _b.Slice(padded, 0, j, frames);
            }

            input = _b.Concat(1, windows);
        }

        return _b.Emit1("Conv", bias is null ? [input, weight] : [input, weight, bias], attributes);
    }

    /// <summary>RMS normalization over the channels of every pixel without a scale: x / √(mean(x²) + ε).</summary>
    private Value RmsNorm(Value x)
    {
        var meanSquare = _b.Emit1("Reduce", [_b.Unary("Square", x)], Attributes.From(("mode", "mean"), ("axes", new[] { 1L }), ("keepdims", true)));
        return _b.Mul(x, _b.Unary("Rsqrt", _b.Scale(meanSquare, 1f, RmsEps)));
    }

    /// <summary>Layer normalization over the channels of every pixel with <c>{prefix}.weight</c> / <c>{prefix}.bias</c>.</summary>
    private Value LayerNorm(Value x, string prefix)
    {
        long channels = x.Shape[1].Fixed;
        var mean = _b.Emit1("Reduce", [x], Attributes.From(("mode", "mean"), ("axes", new[] { 1L }), ("keepdims", true)));
        var centered = _b.Sub(x, mean);
        var variance = _b.Emit1("Reduce", [_b.Unary("Square", centered)], Attributes.From(("mode", "mean"), ("axes", new[] { 1L }), ("keepdims", true)));
        var normed = _b.Mul(centered, _b.Unary("Rsqrt", _b.Scale(variance, 1f, LayerNormEps)));
        return _b.Add(_b.Mul(normed, _b.Reshape(_b.Weight(prefix + ".weight"), 1, channels, 1, 1)), _b.Reshape(_b.Weight(prefix + ".bias"), 1, channels, 1, 1));
    }

    /// <summary>
    /// The embedded decode timestep of a block (<c>{prefix}.timestep_embedder.linear_1|2</c> over its 256-channel sinusoids, diffusers'
    /// <c>PixArtAlphaCombinedTimestepSizeEmbeddings</c> without size conditions), or null for decoders without timestep conditioning.
    /// </summary>
    private Value? TimeEmbedding(string prefix)
    {
        if (_timestep is null || !_b.HasWeight(prefix + ".timestep_embedder.linear_1.weight"))
        {
            return null;
        }

        const int half = 128;
        var frequencies = new float[half];
        for (int i = 0; i < half; i++)
        {
            frequencies[i] = (float)Math.Exp(-Math.Log(10000) * i / half);
        }

        var table = _b.Constant(prefix + ".frequencies", DataType.F32, new Shape(1, half), System.Runtime.InteropServices.MemoryMarshal.AsBytes(frequencies.AsSpan()).ToArray());
        var arguments = _b.Mul(_b.Reshape(_timestep, 1, 1), table);
        var sinusoids = _b.Concat(1, _b.Unary("Cos", arguments), _b.Unary("Sin", arguments));
        string e = prefix + ".timestep_embedder.";
        return _b.Linear(_b.Silu(_b.Linear(sinusoids, e + "linear_1.weight", e + "linear_1.bias")), e + "linear_2.weight", e + "linear_2.bias");
    }

    private Value ResidualBlock(Value x, string prefix, Value? time = null)
    {
        Value? shift1 = null, scale1 = null, shift2 = null, scale2 = null;
        if (time is not null && _b.HasWeight(prefix + ".scale_shift_table"))
        {
            long channels = x.Shape[1].Fixed;
            var table = _b.Add(_b.Reshape(_b.Weight(prefix + ".scale_shift_table"), 4, channels), _b.Reshape(time, 4, channels));
            Value Row(int index) => _b.Reshape(_b.Slice(table, 0, index, 1), 1, channels, 1, 1);
            (shift1, scale1, shift2, scale2) = (Row(0), Row(1), Row(2), Row(3));
        }

        var normed = RmsNorm(x);
        if (shift1 is not null)
        {
            normed = _b.Add(_b.Mul(normed, _b.Scale(scale1!, 1f, 1f)), shift1);
        }

        var h = Conv(_b.Silu(normed), prefix + ".conv1");
        normed = RmsNorm(h);
        if (shift2 is not null)
        {
            normed = _b.Add(_b.Mul(normed, _b.Scale(scale2!, 1f, 1f)), shift2);
        }

        h = Conv(_b.Silu(normed), prefix + ".conv2");
        var shortcut = x;
        if (_b.HasWeight(prefix + ".norm3.weight"))
        {
            shortcut = LayerNorm(shortcut, prefix + ".norm3");
        }

        if (_b.HasWeight(prefix + ".conv_shortcut.conv.weight"))
        {
            shortcut = Conv(shortcut, prefix + ".conv_shortcut");
        }

        return _b.Add(h, shortcut);
    }

    /// <summary>
    /// The 2×2×2 depth-to-space upsampler: a convolution to 8·C / factor channels shuffled into 2 frames and 2×2 pixels; the first new frame
    /// is dropped. With <paramref name="residual"/> the input, shuffled the same way and its channels repeated 8 / factor times, is added.
    /// </summary>
    private Value Upsample(Value x, string prefix, bool residual, int factor)
    {
        var y = DepthToSpace(Conv(x, prefix + ".conv"));
        if (residual)
        {
            var shuffled = DepthToSpace(x);
            int repeats = 8 / factor;
            y = _b.Add(y, repeats == 1 ? shuffled : _b.Concat(1, [.. Enumerable.Repeat(shuffled, repeats)]));
        }

        return y;
    }

    /// <summary>Frames [T, 8·C, H, W] → [2T − 1, C, 2H, 2W]: channels (c, t, h, w) shuffled into 2 frames and 2×2 pixels, the first frame dropped.</summary>
    private Value DepthToSpace(Value y)
    {
        long frames = y.Shape[0].Fixed;
        long height = y.Shape[2].Fixed;
        long width = y.Shape[3].Fixed;
        long channels = y.Shape[1].Fixed / 8;
        var byFrame = _b.Contiguous(_b.Permute(_b.Reshape(y, frames, channels, 2, 4 * height * width), 0, 2, 1, 3));
        var pixels = _b.Reshape(byFrame, frames * 2, channels, 2, 2, height, width);
        var shuffled = _b.Reshape(_b.Contiguous(_b.Permute(pixels, 0, 1, 4, 2, 5, 3)), frames * 2, channels, height * 2, width * 2);
        return _b.Contiguous(_b.Slice(shuffled, 0, 1, frames * 2 - 1));
    }

    /// <summary>Frames [n, 3·p·p, h, w] → [n, 3, h·p, w·p], channels ordered (channel, column in patch, row in patch).</summary>
    private Value Unpatchify(Value x)
    {
        int p = _patchSize;
        long frames = x.Shape[0].Fixed;
        long height = x.Shape[2].Fixed;
        long width = x.Shape[3].Fixed;
        var split = _b.Reshape(x, frames, 3, p, p, height, width);
        var ordered = _b.Contiguous(_b.Permute(split, 0, 1, 4, 3, 5, 2));
        return _b.Reshape(ordered, frames, 3, height * p, width * p);
    }

    private Value ChannelWeight(string name, int channels) => _b.Reshape(_b.Weight(name), 1, channels, 1, 1);
}
