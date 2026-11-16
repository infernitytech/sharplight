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
/// </summary>
/// <remarks>
/// Canonical weights (diffusers names under <c>vae.</c>): <c>latents_mean</c> and <c>latents_std</c> f32 [C]; convolutions
/// <c>{name}.conv.weight</c> [O, kt·C, kh, kw] with <c>{name}.conv.bias</c>; <c>{name}.norm3.weight|bias</c> of the shortcut layer norms.
/// </remarks>
internal sealed class LtxVaeGraphBuilder
{
    private const string P = "vae.";
    private const float RmsEps = 1e-8f;
    private const float LayerNormEps = 1e-6f;

    private readonly GraphBuilder _b;
    private readonly int _latentChannels;
    private readonly int _patchSize;

    private LtxVaeGraphBuilder(Func<string, WeightInfo?> weights, int latentChannels, int patchSize)
    {
        _b = new GraphBuilder(VideoDiffusionGraphs.VaeDecode, weights);
        _latentChannels = latentChannels;
        _patchSize = patchSize;
    }

    /// <summary>Builds the whole-video decoder graph.</summary>
    /// <param name="weights">Canonical weight lookup.</param>
    /// <param name="latentChannels">Latent channels.</param>
    /// <param name="patchSize">Pixel patch size of the output (4).</param>
    /// <param name="latentFrames">Latent frames.</param>
    /// <param name="latentHeight">Latent height.</param>
    /// <param name="latentWidth">Latent width.</param>
    public static ComputeGraph BuildDecoder(Func<string, WeightInfo?> weights, int latentChannels, int patchSize, long latentFrames, long latentHeight, long latentWidth)
        => new LtxVaeGraphBuilder(weights, latentChannels, patchSize).Decoder(latentFrames, latentHeight, latentWidth);

    private ComputeGraph Decoder(long frames, long height, long width)
    {
        int c = _latentChannels;
        var latent = _b.Input(VideoDiffusionGraphs.Latent, InputRole.Other, DataType.F32, new Shape(c, frames, height, width));
        var x = _b.Contiguous(_b.Permute(latent, 1, 0, 2, 3));
        x = _b.Add(_b.Mul(x, ChannelWeight(P + "latents_std", c)), ChannelWeight(P + "latents_mean", c));
        x = Conv(x, P + "decoder.conv_in");
        for (int r = 0; _b.HasWeight($"{P}decoder.mid_block.resnets.{r}.conv1.conv.weight"); r++)
        {
            x = ResidualBlock(x, $"{P}decoder.mid_block.resnets.{r}");
        }

        for (int i = 0; _b.HasWeight($"{P}decoder.up_blocks.{i}.resnets.0.conv1.conv.weight"); i++)
        {
            using var _ = _b.Group($"up.{i}");
            string p = $"{P}decoder.up_blocks.{i}";
            if (_b.HasWeight(p + ".conv_in.conv1.conv.weight"))
            {
                x = ResidualBlock(x, p + ".conv_in");
            }

            if (_b.HasWeight(p + ".upsamplers.0.conv.conv.weight"))
            {
                x = Upsample(x, p + ".upsamplers.0");
            }

            for (int r = 0; _b.HasWeight($"{p}.resnets.{r}.conv1.conv.weight"); r++)
            {
                x = ResidualBlock(x, $"{p}.resnets.{r}");
            }
        }

        x = Conv(_b.Silu(RmsNorm(x)), P + "decoder.conv_out");
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

    private Value ResidualBlock(Value x, string prefix)
    {
        var h = Conv(_b.Silu(RmsNorm(x)), prefix + ".conv1");
        h = Conv(_b.Silu(RmsNorm(h)), prefix + ".conv2");
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

    /// <summary>The 2×2×2 depth-to-space upsampler: a convolution to 8·C channels shuffled into 2 frames and 2×2 pixels; the first new frame is dropped.</summary>
    private Value Upsample(Value x, string prefix)
    {
        long frames = x.Shape[0].Fixed;
        long height = x.Shape[2].Fixed;
        long width = x.Shape[3].Fixed;
        var y = Conv(x, prefix + ".conv");
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
