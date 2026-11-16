using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;
using Sharplight.VideoGeneration;

namespace Sharplight.Architectures;

/// <summary>
/// Builds the decoder of a HunyuanVideo VAE (<c>AutoencoderKLHunyuanVideo</c>) as one graph over a latent video
/// (<see cref="VideoDiffusionGraphs.VaeDecode"/>), on frames-as-batch activations [T, C, H, W]: causal 3-D convolutions (the first frame
/// replicated twice in front, edge pixels replicated spatially; the kernel's time taps folded into its input channels), group norms whose
/// statistics span all frames, a mid block with single-head attention in which every pixel sees the pixels of its own and earlier frames,
/// residual up blocks with nearest upsamplers (×2 spatially; ×2 in time for the middle blocks, the first frame not repeated), and the
/// output norm and convolution; the latent scaling factor is undone first and the frames are returned scaled to [0, 1] (not clamped).
/// </summary>
/// <remarks>
/// Canonical weights (diffusers names under <c>vae.</c>): <c>post_quant_conv.weight</c> [C, C, 1, 1] and bias; convolutions
/// <c>decoder.{name}.conv.weight</c> [O, kt·C, kh, kw]; group norms <c>{name}.weight|bias</c>; the mid attention's <c>group_norm</c>,
/// <c>to_q|to_k|to_v|to_out.0</c>.
/// </remarks>
internal sealed class HunyuanVideoVaeGraphBuilder
{
    private const string P = "vae.";
    private const float Eps = 1e-6f;

    private readonly GraphBuilder _b;
    private readonly int _groups;

    private HunyuanVideoVaeGraphBuilder(Func<string, WeightInfo?> weights, int groups)
    {
        _b = new GraphBuilder(VideoDiffusionGraphs.VaeDecode, weights);
        _groups = groups;
    }

    /// <summary>Builds the decoder graph for one latent video size.</summary>
    /// <param name="weights">Canonical weight lookup.</param>
    /// <param name="latentChannels">Latent channels.</param>
    /// <param name="scalingFactor">The latent scaling factor (latents are divided by it).</param>
    /// <param name="groups">Group-norm groups.</param>
    /// <param name="temporalCompression">Temporal compression (4).</param>
    /// <param name="spatialCompression">Spatial compression (8).</param>
    /// <param name="latentFrames">Latent frames.</param>
    /// <param name="latentHeight">Latent height.</param>
    /// <param name="latentWidth">Latent width.</param>
    public static ComputeGraph BuildDecoder(Func<string, WeightInfo?> weights, int latentChannels, float scalingFactor, int groups, int temporalCompression,
        int spatialCompression, long latentFrames, long latentHeight, long latentWidth)
        => new HunyuanVideoVaeGraphBuilder(weights, groups).Decoder(latentChannels, scalingFactor, temporalCompression, spatialCompression, latentFrames, latentHeight, latentWidth);

    private ComputeGraph Decoder(int channels, float scalingFactor, int temporalCompression, int spatialCompression, long frames, long height, long width)
    {
        var latent = _b.Input(VideoDiffusionGraphs.Latent, InputRole.Other, DataType.F32, new Shape(channels, frames, height, width));
        var x = _b.Scale(_b.Contiguous(_b.Permute(latent, 1, 0, 2, 3)), 1f / scalingFactor);
        x = Conv(x, P + "post_quant_conv", causal: false);
        x = Conv(x, P + "decoder.conv_in.conv");
        using (_b.Group("mid"))
        {
            x = ResidualBlock(x, P + "decoder.mid_block.resnets.0");
            if (_b.HasWeight(P + "decoder.mid_block.attentions.0.to_q.weight"))
            {
                x = Attention(x, P + "decoder.mid_block.attentions.0");
            }

            x = ResidualBlock(x, P + "decoder.mid_block.resnets.1");
        }

        int blocks = 0;
        while (_b.HasWeight($"{P}decoder.up_blocks.{blocks}.resnets.0.conv1.conv.weight"))
        {
            blocks++;
        }

        int spatialLayers = (int)Math.Log2(spatialCompression);
        int temporalLayers = (int)Math.Log2(temporalCompression);
        for (int i = 0; i < blocks; i++)
        {
            using var _ = _b.Group($"up.{i}");
            string p = $"{P}decoder.up_blocks.{i}";
            for (int r = 0; _b.HasWeight($"{p}.resnets.{r}.conv1.conv.weight"); r++)
            {
                x = ResidualBlock(x, $"{p}.resnets.{r}");
            }

            if (_b.HasWeight($"{p}.upsamplers.0.conv.conv.weight"))
            {
                // diffusers: spatial ×2 for the first log2(spatial) blocks, temporal ×2 for the log2(temporal) blocks before the last.
                bool temporal = i >= blocks - 1 - temporalLayers && i != blocks - 1;
                bool spatial = i < spatialLayers;
                x = Conv(Upsample(x, temporal, spatial), $"{p}.upsamplers.0.conv.conv");
            }
        }

        x = Conv(_b.Silu(GroupNorm(x, P + "decoder.conv_norm_out")), P + "decoder.conv_out.conv");
        // Not clamped: temporal decode tiles are blended before the pixels are clamped, as diffusers does.
        _b.Output(VideoDiffusionGraphs.Frames, _b.Scale(x, 0.5f, 0.5f));
        return _b.Build();
    }

    /// <summary>
    /// A causal 3-D convolution over frames [T, C, H, W]: the first frame replicated kt − 1 times in front, edge pixels replicated
    /// spatially, the windows of kt frames stacked along the channels (time-tap-major) for the folded 2-D kernel.
    /// </summary>
    private Value Conv(Value x, string prefix, bool causal = true)
    {
        var weight = _b.Weight(prefix + ".weight");
        var bias = _b.TryWeight(prefix + ".bias");
        long channels = x.Shape[1].Fixed;
        long taps = weight.Shape[1].Fixed / channels;
        long half = weight.Shape[2].Fixed / 2;
        var input = x;
        if (causal && taps > 1)
        {
            long frames = x.Shape[0].Fixed;
            var first = _b.Slice(x, 0, 0, 1);
            var parts = Enumerable.Repeat(first, (int)(taps - 1)).Append(x).ToArray();
            var padded = _b.Concat(0, parts);
            var windows = new Value[taps];
            for (int j = 0; j < taps; j++)
            {
                windows[j] = _b.Slice(padded, 0, j, frames);
            }

            input = _b.Concat(1, windows);
        }

        if (half > 0)
        {
            input = ReplicatePad(input, half);
        }

        var attributes = Attributes.From(("strides", new[] { 1L, 1L }), ("pads", new[] { 0L, 0L, 0L, 0L }));
        return _b.Emit1("Conv", bias is null ? [input, weight] : [input, weight, bias], attributes);
    }

    /// <summary>Pads [N, C, H, W] by <paramref name="pad"/> edge-replicated pixels on every side.</summary>
    private Value ReplicatePad(Value x, long pad)
    {
        long height = x.Shape[2].Fixed;
        var top = _b.Slice(x, 2, 0, 1);
        var bottom = _b.Slice(x, 2, height - 1, 1);
        x = _b.Concat(2, [.. Enumerable.Repeat(top, (int)pad), x, .. Enumerable.Repeat(bottom, (int)pad)]);
        long width = x.Shape[3].Fixed;
        var left = _b.Slice(x, 3, 0, 1);
        var right = _b.Slice(x, 3, width - 1, 1);
        return _b.Concat(3, [.. Enumerable.Repeat(left, (int)pad), x, .. Enumerable.Repeat(right, (int)pad)]);
    }

    /// <summary>Group normalization of frames [T, C, H, W] with statistics over the group's channels in all frames (a 3-D group norm).</summary>
    private Value GroupNorm(Value x, string prefix)
    {
        long frames = x.Shape[0].Fixed;
        long channels = x.Shape[1].Fixed;
        long height = x.Shape[2].Fixed;
        long width = x.Shape[3].Fixed;
        var volume = _b.Reshape(_b.Contiguous(_b.Permute(_b.Reshape(x, frames, channels, height * width), 1, 0, 2)), 1, channels, frames, height * width);
        var normed = DiffusionBlocks.GroupNorm(_b, volume, prefix, _groups, Eps);
        return _b.Reshape(_b.Contiguous(_b.Permute(_b.Reshape(normed, channels, frames, height * width), 1, 0, 2)), frames, channels, height, width);
    }

    private Value ResidualBlock(Value x, string prefix)
    {
        var h = Conv(_b.Silu(GroupNorm(x, prefix + ".norm1")), prefix + ".conv1.conv");
        h = Conv(_b.Silu(GroupNorm(h, prefix + ".norm2")), prefix + ".conv2.conv");
        var shortcut = _b.HasWeight(prefix + ".conv_shortcut.conv.weight") ? Conv(x, prefix + ".conv_shortcut.conv") : x;
        return _b.Add(h, shortcut);
    }

    /// <summary>
    /// Single-head self-attention over all pixels of all frames, each pixel seeing the pixels of its own and the earlier frames (key spans),
    /// behind a group norm, with a residual connection.
    /// </summary>
    private Value Attention(Value x, string prefix)
    {
        long frames = x.Shape[0].Fixed;
        long channels = x.Shape[1].Fixed;
        long height = x.Shape[2].Fixed;
        long width = x.Shape[3].Fixed;
        long pixels = height * width;
        long rows = frames * pixels;
        var normed = GroupNorm(x, prefix + ".group_norm");
        var tokens = _b.Reshape(_b.Contiguous(_b.Permute(_b.Reshape(normed, frames, channels, pixels), 0, 2, 1)), rows, channels);
        var q = _b.Reshape(_b.Linear(tokens, prefix + ".to_q.weight", prefix + ".to_q.bias"), rows, 1, channels);
        var k = _b.Reshape(_b.Linear(tokens, prefix + ".to_k.weight", prefix + ".to_k.bias"), rows, 1, channels);
        var v = _b.Reshape(_b.Linear(tokens, prefix + ".to_v.weight", prefix + ".to_v.bias"), rows, 1, channels);
        var spans = new int[rows * 2];
        for (long t = 0; t < rows; t++)
        {
            spans[t * 2 + 1] = (int)((t / pixels + 1) * pixels);
        }

        var span = _b.Constant(prefix + ".frame_spans", DataType.I32, new Shape(rows, 2), MemoryMarshal.AsBytes(spans.AsSpan()).ToArray());
        var attended = _b.Emit1("Attention", [q, k, v, span], Attributes.From(("scale", 1f / MathF.Sqrt(channels)), ("causal", false)));
        var projected = _b.Linear(_b.Reshape(attended, rows, channels), prefix + ".to_out.0.weight", prefix + ".to_out.0.bias");
        var back = _b.Reshape(_b.Contiguous(_b.Permute(_b.Reshape(projected, frames, pixels, channels), 0, 2, 1)), frames, channels, height, width);
        return _b.Add(x, back);
    }

    /// <summary>
    /// Nearest upsampling of frames [T, C, H, W]: ×2 spatially when <paramref name="spatial"/>; when <paramref name="temporal"/> every frame
    /// after the first is repeated (T → 1 + 2 (T − 1)).
    /// </summary>
    private Value Upsample(Value x, bool temporal, bool spatial)
    {
        if (spatial)
        {
            long[] size = [x.Shape[2].Fixed * 2, x.Shape[3].Fixed * 2];
            x = _b.Emit1("Resize", [x], Attributes.From(("size", size), ("mode", "nearest"), ("coord", "asymmetric"), ("nearest_mode", "floor")));
        }

        long frames = x.Shape[0].Fixed;
        if (!temporal || frames == 1)
        {
            return x;
        }

        long channels = x.Shape[1].Fixed;
        long height = x.Shape[2].Fixed;
        long width = x.Shape[3].Fixed;
        var rest = _b.Reshape(_b.Slice(x, 0, 1, frames - 1), frames - 1, 1, channels * height * width);
        var doubled = _b.Reshape(_b.Concat(1, rest, rest), 2 * (frames - 1), channels, height, width);
        return _b.Concat(0, _b.Slice(x, 0, 0, 1), doubled);
    }
}
