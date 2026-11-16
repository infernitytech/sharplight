using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.ImageGeneration;
using Sharplight.Models;
using Sharplight.Tensors;
using Sharplight.VideoGeneration;

namespace Sharplight.Architectures;

/// <summary>
/// Builds the graphs of a Wan-style causal 3-D video VAE (Wan 2.1, and the residual Wan 2.2 VAE with 2×2 pixel patching) from
/// generic 2-D ops on frames-as-batch activations [T, C, H, W]: a causal 3-D convolution over the frames of a chunk is one 2-D
/// convolution over the frame windows stacked along the channels (the kernel's time taps folded into its input channels by the
/// importer), with the two frames before the chunk taken from the cache of the previous chunk (zeros before the first). Channel RMS
/// norms, per-frame single-head attention, nearest 2× upsampling with temporal doubling by a (3, 1, 1) convolution, strided
/// downsampling and the pixel-shuffle shortcuts of the residual blocks complete the networks.
/// </summary>
/// <remarks>
/// <para>
/// The decoder streams over latent frames as the reference implementation does (<see cref="VideoDiffusionGraphs"/>): the first graph
/// decodes latent frame 0 into one frame (no temporal upsampling), the next graph each further latent frame into
/// <see cref="WanVaeHyperParameters.TemporalDownscale"/> frames; the encoder graph encodes one image (the first chunk) for
/// image-to-video.
/// </para>
/// <para>
/// Canonical weights (the diffusers <c>AutoencoderKLWan</c> names under <c>vae.</c>): causal convolutions <c>{name}.weight</c>
/// [O, kt·C, kh, kw] (time-tap-major: input channel j·C + c is tap j of channel c; the last tap multiplies the newest frame) with
/// <c>{name}.bias</c>; norm scales <c>{name}.gamma</c> f32 [C]; attention <c>to_qkv</c> [3C, C] and <c>proj</c> [C, C] as matrices.
/// </para>
/// </remarks>
internal sealed class WanVaeGraphBuilder
{
    private const string P = "vae.";

    private readonly GraphBuilder _b;
    private readonly WanVaeHyperParameters _hp;
    private readonly bool _first;
    private readonly bool _emitCaches;
    private readonly bool _image;

    private WanVaeGraphBuilder(string name, Func<string, WeightInfo?> weights, WanVaeHyperParameters hp, bool first, bool image = false)
    {
        _b = new GraphBuilder(name, weights);
        _image = image;
        _emitCaches = !image;
        _hp = hp;
        _first = first;
    }

    /// <summary>
    /// Builds the image decoder (<see cref="DiffusionGraphs.VaeDecode"/>: one latent frame, latents [1, z, h, w] in the pipeline's
    /// normalized space → image [1, 3, H, W] in [0, 1]) of an image generator with this VAE (Qwen-Image).
    /// </summary>
    /// <param name="weights">Canonical weight lookup.</param>
    /// <param name="hp">The VAE hyper-parameters.</param>
    /// <param name="latentHeight">Latent height.</param>
    /// <param name="latentWidth">Latent width.</param>
    public static ComputeGraph BuildImageDecoder(Func<string, WeightInfo?> weights, WanVaeHyperParameters hp, long latentHeight, long latentWidth)
        => new WanVaeGraphBuilder(DiffusionGraphs.VaeDecode, weights, hp, first: true, image: true).Decoder(latentHeight, latentWidth);

    /// <summary>
    /// Builds the image encoder (<see cref="DiffusionGraphs.VaeEncode"/>: image [1, 3, H, W] in [0, 1] → normalized latents
    /// [1, z, h, w]) of an image generator with this VAE.
    /// </summary>
    /// <param name="weights">Canonical weight lookup.</param>
    /// <param name="hp">The VAE hyper-parameters.</param>
    /// <param name="height">Image height in pixels.</param>
    /// <param name="width">Image width in pixels.</param>
    public static ComputeGraph BuildImageEncoder(Func<string, WeightInfo?> weights, WanVaeHyperParameters hp, long height, long width)
        => new WanVaeGraphBuilder(DiffusionGraphs.VaeEncode, weights, hp, first: true, image: true).Encoder(height, width, 1);

    /// <summary>Builds a decoder graph of one latent frame.</summary>
    /// <param name="weights">Canonical weight lookup.</param>
    /// <param name="hp">The VAE hyper-parameters.</param>
    /// <param name="latentHeight">Latent height.</param>
    /// <param name="latentWidth">Latent width.</param>
    /// <param name="first">The first latent frame (<see cref="VideoDiffusionGraphs.VaeDecodeFirst"/>) or a further one.</param>
    public static ComputeGraph BuildDecoder(Func<string, WeightInfo?> weights, WanVaeHyperParameters hp, long latentHeight, long latentWidth, bool first)
        => new WanVaeGraphBuilder(first ? VideoDiffusionGraphs.VaeDecodeFirst : VideoDiffusionGraphs.VaeDecodeNext, weights, hp, first)
            .Decoder(latentHeight, latentWidth);

    /// <summary>Builds the encoder graph of one image (the first latent frame).</summary>
    /// <param name="weights">Canonical weight lookup.</param>
    /// <param name="hp">The VAE hyper-parameters.</param>
    /// <param name="height">Image height in pixels.</param>
    /// <param name="width">Image width in pixels.</param>
    public static ComputeGraph BuildEncoder(Func<string, WeightInfo?> weights, WanVaeHyperParameters hp, long height, long width)
        => new WanVaeGraphBuilder(VideoDiffusionGraphs.VaeEncode, weights, hp, first: true).Encoder(height, width, 1);

    /// <summary>
    /// Builds the encoder graph of each further chunk of temporal-compression frames (one latent frame), with the causal caches of the
    /// previous chunk (<see cref="VideoDiffusionGraphs.VaeEncodeNext"/>; the Wan 2.1 VAE).
    /// </summary>
    /// <param name="weights">Canonical weight lookup.</param>
    /// <param name="hp">The VAE hyper-parameters.</param>
    /// <param name="height">Frame height in pixels.</param>
    /// <param name="width">Frame width in pixels.</param>
    public static ComputeGraph BuildEncoderNext(Func<string, WeightInfo?> weights, WanVaeHyperParameters hp, long height, long width)
        => new WanVaeGraphBuilder(VideoDiffusionGraphs.VaeEncodeNext, weights, hp, first: false).Encoder(height, width, hp.TemporalDownscale);

    private ComputeGraph Decoder(long latentHeight, long latentWidth)
    {
        int z = _hp.LatentChannels;
        var x = _image
            ? _b.Input(DiffusionGraphs.Latent, InputRole.Media, DataType.F32, new Shape(1, z, latentHeight, latentWidth))
            : _b.Reshape(_b.Input(VideoDiffusionGraphs.Latent, InputRole.Other, DataType.F32, new Shape(z, 1, latentHeight, latentWidth)), 1, z, latentHeight, latentWidth);
        // The pipeline's un-normalization: latents · std + mean per channel.
        x = _b.Add(_b.Mul(x, ChannelConstant("latents_std", _hp.LatentsStd)), ChannelConstant("latents_mean", _hp.LatentsMean));
        x = CausalConv(x, P + "post_quant_conv");
        x = CausalConv(x, P + "decoder.conv_in");
        x = MidBlock(x, P + "decoder.mid_block");
        int levels = _hp.ChannelMultipliers.Length;
        for (int i = 0; i < levels; i++)
        {
            using var _ = _b.Group($"up.{i}");
            bool up = i != levels - 1;
            bool temporal = up && _hp.TemporalDownsample[levels - 2 - i];
            x = _hp.Residual ? ResidualUpBlock(x, $"{P}decoder.up_blocks.{i}", up, temporal) : UpBlock(x, $"{P}decoder.up_blocks.{i}", up, temporal);
        }

        x = CausalConv(_b.Silu(ChannelNorm(x, P + "decoder.norm_out")), P + "decoder.conv_out");
        if (_hp.PatchSize > 1)
        {
            x = Unpatchify(x);
        }

        var clamped = _b.Emit1("Clip", [x], Attributes.From(("min", -1f), ("max", 1f)));
        _b.Output(_image ? DiffusionGraphs.Image : VideoDiffusionGraphs.Frames, _b.Scale(clamped, 0.5f, 0.5f));
        return _b.Build();
    }

    private ComputeGraph Encoder(long height, long width, long frames)
    {
        int z = _hp.LatentChannels;
        Value x;
        if (_image)
        {
            var image = _b.Input(DiffusionGraphs.Image, InputRole.Media, DataType.F32, new Shape(1, 3, height, width));
            x = _b.Reshape(_b.Scale(image, 2f, -1f), 1, 3, height, width);
        }
        else if (_first)
        {
            var image = _b.Input(VideoDiffusionGraphs.Image, InputRole.Media, DataType.F32, new Shape(3, height, width));
            x = _b.Reshape(_b.Scale(image, 2f, -1f), 1, 3, height, width);
        }
        else
        {
            var video = _b.Input(VideoDiffusionGraphs.Frames, InputRole.Media, DataType.F32, new Shape(3, frames, height, width));
            x = _b.Contiguous(_b.Permute(_b.Scale(video, 2f, -1f), 1, 0, 2, 3));
        }

        if (_hp.PatchSize > 1)
        {
            x = Patchify(x);
        }

        x = CausalConv(x, P + "encoder.conv_in");
        int levels = _hp.ChannelMultipliers.Length;
        int index = 0;
        for (int i = 0; i < levels; i++)
        {
            using var _ = _b.Group($"down.{i}");
            bool down = i != levels - 1;
            if (_hp.Residual)
            {
                x = ResidualDownBlock(x, $"{P}encoder.down_blocks.{i}", down, down && _hp.TemporalDownsample[i]);
                continue;
            }

            for (int r = 0; r < _hp.ResidualBlocks; r++)
            {
                x = ResidualBlock(x, $"{P}encoder.down_blocks.{index++}");
            }

            if (down)
            {
                x = Downsample(x, $"{P}encoder.down_blocks.{index++}", _hp.TemporalDownsample[i]);
            }
        }

        x = MidBlock(x, P + "encoder.mid_block");
        x = CausalConv(_b.Silu(ChannelNorm(x, P + "encoder.norm_out")), P + "encoder.conv_out");
        x = CausalConv(x, P + "quant_conv");
        long latentHeight = x.Shape[2].Fixed;
        long latentWidth = x.Shape[3].Fixed;
        if (x.Shape[0].Fixed != 1)
        {
            throw new InvalidOperationException($"The encoder chunk produced {x.Shape[0].Fixed} latent frames instead of one.");
        }

        // The mean half of the latent distribution, normalized: (μ − mean) / std.
        var mean = _b.Slice(x, 1, 0, z);
        var normalized = _b.Mul(_b.Sub(mean, ChannelConstant("latents_mean", _hp.LatentsMean)), ChannelConstant("latents_inv_std", [.. _hp.LatentsStd.Select(s => 1f / s)]));
        _b.Output(_image ? DiffusionGraphs.Latent : VideoDiffusionGraphs.Latent,
            _image ? _b.Reshape(normalized, 1, z, latentHeight, latentWidth) : _b.Reshape(normalized, z, latentHeight, latentWidth));
        return _b.Build();
    }

    /// <summary>
    /// A causal 3-D convolution over the frames of this chunk [T, C, H, W] (spatial "same" padding). Kernels with one time tap are a
    /// plain 2-D convolution; with kt taps each output frame sees the kt − 1 frames before it, taken from the cache input of the
    /// previous chunk (the next-chunk graph) or zeros (the first chunk). The kt − 1 newest input frames become the cache output.
    /// </summary>
    private Value CausalConv(Value x, string prefix)
    {
        var weight = _b.Weight(prefix + ".weight");
        long channels = x.Shape[1].Fixed;
        long taps = weight.Shape[1].Fixed / channels;
        if (taps == 1)
        {
            return DiffusionBlocks.Conv(_b, x, prefix);
        }

        long frames = x.Shape[0].Fixed;
        string cacheName = prefix[P.Length..];
        Value window;
        if (_first)
        {
            var zeros = _b.Scale(x, 0f);
            var parts = new Value[taps];
            for (int j = 0; j < taps - 1; j++)
            {
                parts[j] = zeros;
            }

            parts[taps - 1] = x;
            window = _b.Concat(1, parts);
            if (_emitCaches)
            {
                _b.Output(VideoDiffusionGraphs.CacheOutPrefix + cacheName, x);
            }
        }
        else
        {
            var cache = _b.Input(VideoDiffusionGraphs.CacheInPrefix + cacheName, InputRole.Other, DataType.F32,
                new Shape(taps - 1, channels, x.Shape[2].Fixed, x.Shape[3].Fixed));
            var full = _b.Concat(0, cache, x);
            var parts = new Value[taps];
            for (int j = 0; j < taps; j++)
            {
                parts[j] = _b.Slice(full, 0, j, frames);
            }

            window = _b.Concat(1, parts);
            _b.Output(VideoDiffusionGraphs.CacheOutPrefix + cacheName, _b.Contiguous(_b.Slice(full, 0, frames, taps - 1)));
        }

        long half = weight.Shape[2].Fixed / 2;
        var attributes = Attributes.From(("strides", new[] { 1L, 1L }), ("pads", new[] { half, half, half, half }));
        var bias = _b.TryWeight(prefix + ".bias");
        return _b.Emit1("Conv", bias is null ? [window, weight] : [window, weight, bias], attributes);
    }

    /// <summary>RMS normalization over the channels of every pixel: x / ‖x‖ · √C · γ (the reference's <c>F.normalize</c> form).</summary>
    private Value ChannelNorm(Value x, string prefix)
    {
        long channels = x.Shape[1].Fixed;
        var norm = _b.Emit1("Reduce", [x], Attributes.From(("mode", "l2"), ("axes", new[] { 1L }), ("keepdims", true)));
        var inverse = _b.Unary("Recip", _b.Emit1("Clip", [norm], Attributes.From(("min", 1e-12f), ("max", float.MaxValue))));
        var gamma = _b.Scale(_b.Reshape(_b.Weight(prefix + ".gamma"), 1, channels, 1, 1), MathF.Sqrt(channels));
        var y = _b.Mul(_b.Mul(x, inverse), gamma);
        return _b.TryWeight(prefix + ".beta") is { } beta ? _b.Add(y, _b.Reshape(beta, 1, channels, 1, 1)) : y;
    }

    private Value ResidualBlock(Value x, string prefix)
    {
        var shortcut = _b.HasWeight(prefix + ".conv_shortcut.weight") ? CausalConv(x, prefix + ".conv_shortcut") : x;
        var h = CausalConv(_b.Silu(ChannelNorm(x, prefix + ".norm1")), prefix + ".conv1");
        h = CausalConv(_b.Silu(ChannelNorm(h, prefix + ".norm2")), prefix + ".conv2");
        return _b.Add(h, shortcut);
    }

    /// <summary>Single-head self-attention over the pixels of each frame.</summary>
    private Value AttentionBlock(Value x, string prefix)
    {
        long frames = x.Shape[0].Fixed;
        long channels = x.Shape[1].Fixed;
        long height = x.Shape[2].Fixed;
        long width = x.Shape[3].Fixed;
        long pixels = height * width;
        var normed = ChannelNorm(x, prefix + ".norm");
        var rows = _b.Reshape(_b.Contiguous(_b.Permute(_b.Reshape(normed, frames, channels, pixels), 0, 2, 1)), frames * pixels, channels);
        var qkv = _b.Linear(rows, prefix + ".to_qkv.weight", prefix + ".to_qkv.bias");
        Value Head(int index) => _b.Reshape(_b.Contiguous(_b.Slice(qkv, 1, index * channels, channels)), frames * pixels, 1, channels);
        var attributes = Attributes.From(("scale", 1f / MathF.Sqrt(channels)), ("causal", false), (BatchedAttention.RowsAttribute, pixels),
            (BatchedAttention.KeysAttribute, pixels));
        var attended = _b.Emit1("Attention", [Head(0), Head(1), Head(2)], attributes);
        var projected = _b.Linear(_b.Reshape(attended, frames * pixels, channels), prefix + ".proj.weight", prefix + ".proj.bias");
        var map = _b.Contiguous(_b.Permute(_b.Reshape(projected, frames, pixels, channels), 0, 2, 1));
        return _b.Add(x, _b.Reshape(map, frames, channels, height, width));
    }

    private Value MidBlock(Value x, string prefix)
    {
        x = ResidualBlock(x, prefix + ".resnets.0");
        x = AttentionBlock(x, prefix + ".attentions.0");
        return ResidualBlock(x, prefix + ".resnets.1");
    }

    /// <summary>A Wan 2.1 decoder level: residual blocks, then 2× (and, temporal, frame-doubling) upsampling.</summary>
    private Value UpBlock(Value x, string prefix, bool up, bool temporal)
    {
        for (int r = 0; r <= _hp.ResidualBlocks; r++)
        {
            x = ResidualBlock(x, $"{prefix}.resnets.{r}");
        }

        return up ? Upsample(x, prefix + ".upsamplers.0", temporal) : x;
    }

    /// <summary>A Wan 2.2 decoder level: residual blocks and upsampling plus the duplicating pixel-shuffle shortcut of the level's input.</summary>
    private Value ResidualUpBlock(Value x, string prefix, bool up, bool temporal)
    {
        var input = x;
        for (int r = 0; r <= _hp.ResidualBlocks; r++)
        {
            x = ResidualBlock(x, $"{prefix}.resnets.{r}");
        }

        if (!up)
        {
            return x;
        }

        x = Upsample(x, prefix + ".upsampler", temporal);
        return _b.Add(x, DuplicateUp(input, x.Shape[1].Fixed, temporal ? 2 : 1));
    }

    /// <summary>A Wan 2.2 encoder level: residual blocks and downsampling plus the averaging pixel-unshuffle shortcut (first chunk).</summary>
    private Value ResidualDownBlock(Value x, string prefix, bool down, bool temporal)
    {
        var input = x;
        for (int r = 0; r < _hp.ResidualBlocks; r++)
        {
            x = ResidualBlock(x, $"{prefix}.resnets.{r}");
        }

        if (down)
        {
            x = Downsample(x, prefix + ".downsampler");
        }

        return _b.Add(x, AverageDown(input, x.Shape[1].Fixed, temporal ? 2 : 1, down ? 2 : 1));
    }

    /// <summary>
    /// Upsampling: in further chunks of a temporal level the (3, 1, 1) causal convolution doubles the channels into two frames per
    /// frame (the first chunk is not doubled), then nearest 2× and a 3×3 convolution.
    /// </summary>
    private Value Upsample(Value x, string prefix, bool temporal)
    {
        if (temporal && !_first)
        {
            long frames = x.Shape[0].Fixed;
            long channels = x.Shape[1].Fixed;
            long height = x.Shape[2].Fixed;
            long width = x.Shape[3].Fixed;
            var doubled = CausalConv(x, prefix + ".time_conv");
            x = _b.Reshape(doubled, frames * 2, channels, height, width);
        }

        return DiffusionBlocks.Upsample(_b, x, prefix + ".resample.1");
    }

    /// <summary>
    /// Strided 2× downsampling (zero padding right and bottom, 3×3 stride-2 convolution). A temporal level halves the frames of a further
    /// chunk with a (3, 1, 1) stride-2 convolution over the previous chunk's last frame and this chunk's frames; the first chunk skips it
    /// and only passes its frame on as the cache.
    /// </summary>
    private Value Downsample(Value x, string prefix, bool temporal = false)
    {
        x = DiffusionBlocks.Conv(_b, x, prefix + ".resample.1", stride: 2, pads: [0, 0, 1, 1]);
        if (!temporal)
        {
            return x;
        }

        string cacheName = prefix[P.Length..] + ".time_conv";
        long frames = x.Shape[0].Fixed;
        long channels = x.Shape[1].Fixed;
        if (_first)
        {
            _b.Output(VideoDiffusionGraphs.CacheOutPrefix + cacheName, x);
            return x;
        }

        var cache = _b.Input(VideoDiffusionGraphs.CacheInPrefix + cacheName, InputRole.Other, DataType.F32, new Shape(1, channels, x.Shape[2].Fixed, x.Shape[3].Fixed));
        var full = _b.Concat(0, cache, x);
        var windows = new Value[frames / 2];
        for (int o = 0; o < windows.Length; o++)
        {
            windows[o] = _b.Concat(1, _b.Slice(full, 0, 2 * o, 1), _b.Slice(full, 0, 2 * o + 1, 1), _b.Slice(full, 0, 2 * o + 2, 1));
        }

        _b.Output(VideoDiffusionGraphs.CacheOutPrefix + cacheName, _b.Contiguous(_b.Slice(x, 0, frames - 1, 1)));
        var stacked = windows.Length == 1 ? windows[0] : _b.Concat(0, windows);
        var weight = _b.Weight(prefix + ".time_conv.weight");
        var bias = _b.TryWeight(prefix + ".time_conv.bias");
        var attributes = Attributes.From(("strides", new[] { 1L, 1L }), ("pads", new[] { 0L, 0L, 0L, 0L }));
        return _b.Emit1("Conv", bias is null ? [stacked, weight] : [stacked, weight, bias], attributes);
    }

    /// <summary>
    /// The decoder shortcut of the residual VAE (<c>DupUp3D</c>): the input channels repeated to out · ft · 2 · 2 and shuffled into ft
    /// frames and 2×2 pixels per input pixel; the first chunk keeps only the last of the ft frames.
    /// </summary>
    private Value DuplicateUp(Value x, long outChannels, int frameFactor)
    {
        long frames = x.Shape[0].Fixed;
        long channels = x.Shape[1].Fixed;
        long height = x.Shape[2].Fixed;
        long width = x.Shape[3].Fixed;
        long expanded = outChannels * frameFactor * 4;
        long repeats = expanded / channels;
        var repeated = _b.Emit1("Expand", [_b.Reshape(x, frames, channels, 1, height * width)], Attributes.From(("shape", new[] { frames, channels, repeats, height * width })));
        var byFrame = _b.Contiguous(_b.Permute(_b.Reshape(repeated, frames, outChannels, frameFactor, 4 * height * width), 0, 2, 1, 3));
        var pixels = _b.Reshape(byFrame, frames * frameFactor, outChannels, 2, 2, height, width);
        var shuffled = _b.Contiguous(_b.Permute(pixels, 0, 1, 4, 2, 5, 3));
        var result = _b.Reshape(shuffled, frames * frameFactor, outChannels, height * 2, width * 2);
        return _first && frameFactor > 1 ? _b.Contiguous(_b.Slice(result, 0, frameFactor - 1, frames * frameFactor - (frameFactor - 1))) : result;
    }

    /// <summary>
    /// The encoder shortcut of the residual VAE (<c>AvgDown3D</c>) for one frame: 2×2 pixels (and ft frames, the missing earlier ones
    /// zero) unshuffled into channels, then averaged in groups down to the output channels.
    /// </summary>
    private Value AverageDown(Value x, long outChannels, int frameFactor, int spatialFactor)
    {
        long channels = x.Shape[1].Fixed;
        long height = x.Shape[2].Fixed / spatialFactor;
        long width = x.Shape[3].Fixed / spatialFactor;
        long cells = (long)spatialFactor * spatialFactor;
        var split = _b.Reshape(x, channels, height, spatialFactor, width, spatialFactor);
        var unshuffled = _b.Reshape(_b.Contiguous(_b.Permute(split, 0, 2, 4, 1, 3)), channels, cells, height * width);
        if (frameFactor > 1)
        {
            var parts = new Value[frameFactor];
            var zeros = _b.Scale(unshuffled, 0f);
            for (int j = 0; j < frameFactor - 1; j++)
            {
                parts[j] = zeros;
            }

            parts[frameFactor - 1] = unshuffled;
            unshuffled = _b.Concat(1, parts);
        }

        long total = channels * cells * frameFactor;
        var grouped = _b.Reshape(unshuffled, outChannels, total / outChannels, height * width);
        var mean = _b.Emit1("Reduce", [grouped], Attributes.From(("mode", "mean"), ("axes", new[] { 1L }), ("keepdims", false)));
        return _b.Reshape(mean, 1, outChannels, height, width);
    }

    /// <summary>Image [1, 3, H, W] → patches [1, 3·p·p, H/p, W/p], channels ordered (channel, column in patch, row in patch).</summary>
    private Value Patchify(Value x)
    {
        int p = _hp.PatchSize;
        long height = x.Shape[2].Fixed / p;
        long width = x.Shape[3].Fixed / p;
        var split = _b.Reshape(x, 3, height, p, width, p);
        var ordered = _b.Contiguous(_b.Permute(split, 0, 4, 2, 1, 3));
        return _b.Reshape(ordered, 1, 3L * p * p, height, width);
    }

    /// <summary>Frames [n, 3·p·p, h, w] → [n, 3, h·p, w·p] (the inverse of <see cref="Patchify"/>).</summary>
    private Value Unpatchify(Value x)
    {
        int p = _hp.PatchSize;
        long frames = x.Shape[0].Fixed;
        long height = x.Shape[2].Fixed;
        long width = x.Shape[3].Fixed;
        var split = _b.Reshape(x, frames, 3, p, p, height, width);
        var ordered = _b.Contiguous(_b.Permute(split, 0, 1, 4, 3, 5, 2));
        return _b.Reshape(ordered, frames, 3, height * p, width * p);
    }

    private Value ChannelConstant(string name, float[] values)
        => _b.Constant(name, DataType.F32, new Shape(1, values.Length, 1, 1), MemoryMarshal.AsBytes(values.AsSpan()).ToArray());
}
