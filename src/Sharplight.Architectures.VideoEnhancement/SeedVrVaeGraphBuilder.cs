using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;
using Sharplight.VideoGeneration;

namespace Sharplight.Architectures.VideoEnhancement;

/// <summary>
/// Builds the encoder (<see cref="SeedVrGraphs.VaeEncode"/>) and decoder (<see cref="SeedVrGraphs.VaeDecode"/>) graphs of SeedVR2's causal
/// video VAE (an SD3-style VAE inflated to 3-D: 16 latent channels, 8× spatial and 4× temporal compression) for one clip size. Activations
/// are frames-first [F, C, H, W]; every group norm and the mid-block attention work per frame, and every convolution is causal in time
/// (<see cref="CausalConvolution"/>), so the graphs run the clip in chunks along time within one graph, each chunk after the previous one
/// (<see cref="GraphBuilder.After"/>), and peak memory follows one chunk: the encoder takes the first frame, then four frames per latent
/// frame; the decoder one latent frame at a time.
/// </summary>
/// <remarks>
/// Canonical weights under <c>vae.</c> (the official names; 3-D kernels folded to [O, kt·C, kh, kw]): <c>encoder.conv_in</c>,
/// <c>encoder.down_blocks.{i}.resnets.{j}.norm1|conv1|norm2|conv2|conv_shortcut</c>, <c>encoder.down_blocks.{i}.downsamplers.0.conv</c>
/// (spatial for the first level, spatial and temporal for the next two), <c>encoder.mid_block.resnets.0|1</c>,
/// <c>encoder.mid_block.attentions.0.group_norm|to_q|to_k|to_v|to_out.0</c>, <c>encoder.conv_norm_out</c>, <c>encoder.conv_out</c> (mean
/// and log-variance); the decoder alike with <c>up_blocks.{i}.upsamplers.0.upscale_conv</c> (1×1 to r·C channels, then depth-to-space by
/// 2×2 spatially and 2 in time for the first two levels) and <c>upsamplers.0.conv</c>.
/// </remarks>
internal sealed class SeedVrVaeGraphBuilder
{
    private const string P = "vae.";

    private readonly GraphBuilder _b;
    private readonly SeedVrVaeHyperParameters _hp;
    private readonly CausalConvolution _conv;

    private SeedVrVaeGraphBuilder(Func<string, WeightInfo?> weights, SeedVrVaeHyperParameters hp, string name)
    {
        _b = new GraphBuilder(name, weights);
        _hp = hp;
        _conv = new CausalConvolution(_b);
    }

    /// <summary>Builds the encoder graph of a clip of <paramref name="frames"/> frames (1 + 4k).</summary>
    /// <param name="weights">Canonical weight lookup.</param>
    /// <param name="hp">The VAE hyper-parameters.</param>
    /// <param name="frames">Frames of the clip.</param>
    /// <param name="height">Height in pixels (a multiple of the spatial compression).</param>
    /// <param name="width">Width in pixels.</param>
    public static ComputeGraph BuildEncoder(Func<string, WeightInfo?> weights, SeedVrVaeHyperParameters hp, int frames, int height, int width)
        => new SeedVrVaeGraphBuilder(weights, hp, SeedVrGraphs.VaeEncode).Encoder(frames, height, width);

    /// <summary>Builds the decoder graph of a latent clip.</summary>
    /// <param name="weights">Canonical weight lookup.</param>
    /// <param name="hp">The VAE hyper-parameters.</param>
    /// <param name="latentFrames">Latent frames.</param>
    /// <param name="latentHeight">Latent height.</param>
    /// <param name="latentWidth">Latent width.</param>
    public static ComputeGraph BuildDecoder(Func<string, WeightInfo?> weights, SeedVrVaeHyperParameters hp, int latentFrames, int latentHeight, int latentWidth)
        => new SeedVrVaeGraphBuilder(weights, hp, SeedVrGraphs.VaeDecode).Decoder(latentFrames, latentHeight, latentWidth);

    private ComputeGraph Encoder(int frames, int height, int width)
    {
        if ((frames - 1) % _hp.TemporalDownscale != 0)
        {
            throw new ArgumentException($"SeedVR2 VAE: {frames} frames is not 1 + {_hp.TemporalDownscale}k.");
        }

        var clip = _b.Input(VideoDiffusionGraphs.Frames, InputRole.Other, DataType.F32, new Shape(frames, 3, height, width));
        int levels = _hp.BlockChannels.Length;
        var means = new List<Value>();
        var deviations = new List<Value>();
        int start = 0;
        for (int chunk = 0; start < frames; chunk++)
        {
            using var _ = _b.Group($"chunk.{chunk}");
            int count = chunk == 0 ? 1 : _hp.TemporalDownscale;
            var chunkFrames = _b.Slice(clip, 0, start, count);
            var x = _conv.Apply(chunk == 0 ? chunkFrames : _b.After(chunkFrames, deviations[^1]), P + "encoder.conv_in");
            start += count;
            for (int i = 0; i < levels; i++)
            {
                string p = $"{P}encoder.down_blocks.{i}";
                for (int j = 0; _b.HasWeight($"{p}.resnets.{j}.conv1.weight"); j++)
                {
                    x = ResidualBlock(x, $"{p}.resnets.{j}");
                }

                if (_b.HasWeight($"{p}.downsamplers.0.conv.weight"))
                {
                    // Zero padding on the bottom and right only, stride 2; the levels with a 3-tap kernel also halve the frames.
                    string conv = $"{p}.downsamplers.0.conv";
                    int taps = (int)(_b.Weight(conv + ".weight").Shape[1].Fixed / x.Shape[1].Fixed);
                    x = _conv.Apply(x, conv, temporalStride: taps > 1 ? 2 : 1, spatialStride: 2, pads: [0, 0, 1, 1]);
                }
            }

            x = MidBlock(x, P + "encoder.mid_block");
            x = _conv.Apply(_b.Silu(GroupNorm(x, P + "encoder.conv_norm_out")), P + "encoder.conv_out");
            int latent = _hp.LatentChannels;
            means.Add(_b.Slice(x, 1, 0, latent));
            var logVariance = _b.Emit1("Clip", [_b.Slice(x, 1, latent, latent)], Attributes.From(("min", -30f), ("max", 20f)));
            deviations.Add(_b.Unary("Exp", _b.Scale(logVariance, 0.5f)));
        }

        _b.Output(SeedVrGraphs.Mean, LatentLayout(means, _hp.ScalingFactor));
        _b.Output(SeedVrGraphs.Deviation, LatentLayout(deviations, _hp.ScalingFactor));
        return _b.Build();
    }

    private ComputeGraph Decoder(int latentFrames, int latentHeight, int latentWidth)
    {
        var latent = _b.Input(VideoDiffusionGraphs.Latent, InputRole.Other, DataType.F32, new Shape(_hp.LatentChannels, latentFrames, latentHeight, latentWidth));
        var framesFirst = _b.Contiguous(_b.Permute(_b.Scale(latent, 1f / _hp.ScalingFactor), 1, 0, 2, 3));
        int levels = _hp.BlockChannels.Length;
        int temporalLevels = (int)Math.Log2(_hp.TemporalDownscale);
        var outputs = new List<Value>();
        for (int chunk = 0; chunk < latentFrames; chunk++)
        {
            using var _ = _b.Group($"chunk.{chunk}");
            var frame = _b.Slice(framesFirst, 0, chunk, 1);
            var x = _conv.Apply(chunk == 0 ? frame : _b.After(frame, outputs[^1]), P + "decoder.conv_in");
            x = MidBlock(x, P + "decoder.mid_block");
            for (int i = 0; i < levels; i++)
            {
                string p = $"{P}decoder.up_blocks.{i}";
                for (int j = 0; _b.HasWeight($"{p}.resnets.{j}.conv1.weight"); j++)
                {
                    x = ResidualBlock(x, $"{p}.resnets.{j}");
                }

                if (_b.HasWeight($"{p}.upsamplers.0.conv.weight"))
                {
                    x = Upsample(x, $"{p}.upsamplers.0", temporal: i < temporalLevels, dropRepeatedFirst: chunk == 0);
                    x = _conv.Apply(x, $"{p}.upsamplers.0.conv");
                }
            }

            outputs.Add(_conv.Apply(_b.Silu(GroupNorm(x, P + "decoder.conv_norm_out")), P + "decoder.conv_out"));
        }

        _b.Output(VideoDiffusionGraphs.Frames, outputs.Count == 1 ? outputs[0] : _b.Concat(0, [.. outputs]));
        return _b.Build();
    }

    /// <summary>Per-chunk latent frames [t, C, h, w] → the scaled latent clip [C, T, h, w].</summary>
    private Value LatentLayout(List<Value> chunks, float scale)
    {
        var all = chunks.Count == 1 ? chunks[0] : _b.Concat(0, [.. chunks]);
        return _b.Scale(_b.Contiguous(_b.Permute(all, 1, 0, 2, 3)), scale);
    }

    /// <summary>GroupNorm, SiLU, conv1, GroupNorm, SiLU, conv2, plus the input (through the 1×1 shortcut when the width changes).</summary>
    private Value ResidualBlock(Value x, string p)
    {
        var h = _conv.Apply(_b.Silu(GroupNorm(x, p + ".norm1")), p + ".conv1");
        h = _conv.Apply(_b.Silu(GroupNorm(h, p + ".norm2")), p + ".conv2");
        var shortcut = _b.HasWeight(p + ".conv_shortcut.weight") ? _conv.Apply(x, p + ".conv_shortcut") : x;
        return _b.Add(shortcut, h);
    }

    /// <summary>Residual block, per-frame single-head attention, residual block.</summary>
    private Value MidBlock(Value x, string p)
    {
        x = ResidualBlock(x, p + ".resnets.0");
        if (_b.HasWeight(p + ".attentions.0.to_q.weight"))
        {
            x = FrameAttention(x, p + ".attentions.0");
        }

        return ResidualBlock(x, p + ".resnets.1");
    }

    /// <summary>GroupNorm, then single-head self-attention among the pixels of each frame, projected and added to the input.</summary>
    private Value FrameAttention(Value x, string p)
    {
        long frames = x.Shape[0].Fixed;
        long channels = x.Shape[1].Fixed;
        long pixels = x.Shape[2].Fixed * x.Shape[3].Fixed;
        var normed = GroupNorm(x, p + ".group_norm");
        var tokens = _b.Reshape(_b.Contiguous(_b.Permute(_b.Reshape(normed, frames, channels, pixels), 0, 2, 1)), frames * pixels, channels);
        var q = _b.Reshape(Linear(tokens, p + ".to_q"), frames * pixels, 1, channels);
        var k = _b.Reshape(Linear(tokens, p + ".to_k"), frames * pixels, 1, channels);
        var v = _b.Reshape(Linear(tokens, p + ".to_v"), frames * pixels, 1, channels);
        var attributes = new List<(string, object)> { ("scale", 1f / MathF.Sqrt(channels)), ("causal", false) };
        if (frames > 1)
        {
            attributes.Add(("sequence_rows", pixels));
            attributes.Add(("sequence_keys", pixels));
        }

        var attended = _b.Emit1("Attention", [q, k, v], Attributes.From([.. attributes]));
        var projected = Linear(_b.Reshape(attended, frames * pixels, channels), p + ".to_out.0");
        var back = _b.Contiguous(_b.Permute(_b.Reshape(projected, frames, pixels, channels), 0, 2, 1));
        return _b.Add(x, _b.Reshape(back, x.Shape[0].Fixed, channels, x.Shape[2].Fixed, x.Shape[3].Fixed));
    }

    /// <summary>
    /// The 1×1 convolution to r·C channels and the depth-to-space rearrangement (channels ordered (row, column, time, channel)) to 2×2 the
    /// pixels and, for temporal levels, 2× the frames; the first chunk keeps one copy of its first frame.
    /// </summary>
    private Value Upsample(Value x, string p, bool temporal, bool dropRepeatedFirst)
    {
        long frames = x.Shape[0].Fixed;
        long channels = x.Shape[1].Fixed;
        long height = x.Shape[2].Fixed;
        long width = x.Shape[3].Fixed;
        int time = temporal ? 2 : 1;
        var expanded = _conv.Apply(x, p + ".upscale_conv");
        var split = _b.Reshape(expanded, frames, 2, 2, time, channels, height, width);
        var ordered = _b.Contiguous(_b.Permute(split, 0, 3, 4, 5, 1, 6, 2));
        var result = _b.Reshape(ordered, frames * time, channels, height * 2, width * 2);
        if (temporal && dropRepeatedFirst)
        {
            long count = frames * time;
            result = count == 2 ? _b.Slice(result, 0, 0, 1) : _b.Concat(0, _b.Slice(result, 0, 0, 1), _b.Slice(result, 0, 2, count - 2));
        }

        return result;
    }

    private Value GroupNorm(Value x, string prefix)
        => _b.Emit1("GroupNorm", [x, _b.Weight(prefix + ".weight"), _b.Weight(prefix + ".bias")], Attributes.From(("groups", (long)_hp.Groups), ("eps", _hp.Eps)));

    private Value Linear(Value x, string module) => _b.Linear(x, module + ".weight", module + ".bias");
}
