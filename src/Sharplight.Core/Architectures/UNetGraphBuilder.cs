using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.ImageGeneration;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Builds the denoiser graph (<see cref="DiffusionGraphs.Denoise"/>) of a latent-diffusion UNet with cross-attention (SD 1.x / 2.x
/// layout): sinusoidal timestep embedding, levels of residual blocks with spatial transformers, a middle block and the mirrored up
/// path with skip connections, for one fixed latent size and a symbolic batch. With a ControlNet it builds
/// <see cref="DiffusionGraphs.DenoiseControl"/> instead: the ControlNet (the UNet's time embedding, input convolution, down path and middle
/// block under <c>controlnet.</c>, the input convolution's output plus the control image's embedding) runs beside the UNet, and its zero
/// convolutions' outputs, scaled per row by <see cref="DiffusionGraphs.ControlScale"/>, are added to the UNet's skip features and to its
/// middle block's output (diffusers' <c>down_block_additional_residuals</c> / <c>mid_block_additional_residual</c>).
/// </summary>
internal sealed class UNetGraphBuilder
{
    private const float TransformerNormEps = 1e-6f;
    private const float LayerNormEps = 1e-5f;
    private const string UNetPrefix = "unet.";

    private readonly GraphBuilder _b;
    private readonly UNetHyperParameters _hp;
    private readonly ControlNetHyperParameters? _control;
    private readonly long _contextRows;
    private Value _context = null!;
    private Value? _pooled;
    private Value? _sizes;

    private UNetGraphBuilder(Func<string, WeightInfo?> weights, UNetHyperParameters hp, long contextRows, ControlNetHyperParameters? control)
    {
        _b = new GraphBuilder(control is null ? DiffusionGraphs.Denoise : DiffusionGraphs.DenoiseControl, weights);
        _hp = hp;
        _contextRows = contextRows;
        _control = control;
    }

    /// <summary>Builds the graph.</summary>
    /// <param name="weights">Canonical weight lookup.</param>
    /// <param name="hp">The UNet hyper-parameters.</param>
    /// <param name="latentHeight">Latent height.</param>
    /// <param name="latentWidth">Latent width.</param>
    /// <param name="contextRows">Tokens of the text conditioning per image.</param>
    /// <param name="control">The ControlNet to run beside the UNet (<see cref="DiffusionGraphs.DenoiseControl"/>), or null for the plain denoiser.</param>
    public static ComputeGraph Build(Func<string, WeightInfo?> weights, UNetHyperParameters hp, long latentHeight, long latentWidth, long contextRows,
        ControlNetHyperParameters? control = null)
        => new UNetGraphBuilder(weights, hp, contextRows, control).Build(latentHeight, latentWidth);

    private ComputeGraph Build(long latentHeight, long latentWidth)
    {
        var batch = Dim.Sym(DiffusionGraphs.BatchSymbol);
        var latent = _b.Input(DiffusionGraphs.Latent, InputRole.Media, DataType.F32, new Shape(batch, _hp.InChannels - _hp.ConditionChannels, latentHeight, latentWidth));
        var timestep = _b.Input(DiffusionGraphs.Timestep, InputRole.Other, DataType.F32, new Shape(batch));
        var context = _b.Input(DiffusionGraphs.Context, InputRole.Other, DataType.F32, new Shape(batch, _contextRows, _hp.ContextDim));
        _context = _b.Emit1("Reshape", [context], Attributes.From(("shape", new[] { 0, (long)_hp.ContextDim }), ("outer_symbol", $"{_contextRows}*{batch}")));
        if (_hp.ClassEmbeddingInput > 0)
        {
            int conditions = _hp.SizeConditions;
            int pooledWidth = _hp.ClassEmbeddingInput - conditions * _hp.AdditionTimeDim;
            _pooled = _b.Input(DiffusionGraphs.Pooled, InputRole.Other, DataType.F32, new Shape(batch, pooledWidth));
            _sizes = _b.Input(DiffusionGraphs.SizeConditions, InputRole.Other, DataType.F32, new Shape(batch, conditions));
        }

        var unetInput = latent;
        if (_hp.ConditionChannels > 0)
        {
            // InstructPix2Pix / inpainting: the image (or mask and masked image) channels follow the noisy latents.
            var condition = _b.Input(DiffusionGraphs.ConditionLatent, InputRole.Media, DataType.F32, new Shape(batch, _hp.ConditionChannels, latentHeight, latentWidth));
            unetInput = _b.Concat(1, latent, condition);
        }

        var (skips, h, time) = Encoder(_hp, UNetPrefix, unetInput, timestep, null);
        if (_control is { } control)
        {
            var (residuals, midResidual) = ControlResiduals(control, latent, timestep, batch, latentHeight, latentWidth);
            for (int i = 0; i < skips.Count; i++)
            {
                skips[i] = _b.Add(skips[i], residuals[i]);
            }

            h = _b.Add(h, midResidual);
        }

        int levels = _hp.BlockChannels.Length;
        int next = skips.Count;
        for (int up = 0; up < levels; up++)
        {
            using var _ = _b.Group($"up.{up}");
            int heads = _hp.HeadCounts[levels - 1 - up];
            for (int i = 0; i <= _hp.LayersPerBlock; i++)
            {
                h = _b.Concat(1, h, skips[--next]);
                h = DiffusionBlocks.ResidualBlock(_b, h, time, $"unet.up.{up}.res.{i}", _hp.NormGroups, _hp.NormEps);
                if (_hp.UpTransformerDepths[up] > 0)
                {
                    h = SpatialTransformer(h, $"unet.up.{up}.attn.{i}", _hp.UpTransformerDepths[up], heads, _hp.NormGroups);
                }
            }

            if (up < levels - 1)
            {
                h = DiffusionBlocks.Upsample(_b, h, $"unet.up.{up}.upsample");
            }
        }

        using (_b.Group("output"))
        {
            h = _b.Silu(DiffusionBlocks.GroupNorm(_b, h, "unet.output_norm", _hp.NormGroups, _hp.NormEps));
            h = DiffusionBlocks.Conv(_b, h, "unet.output_conv");
        }

        _b.Output(DiffusionGraphs.Prediction, h);
        return _b.Build();
    }

    /// <summary>
    /// The down half of a UNet (or of a ControlNet's copy of it, under another <paramref name="prefix"/>): time embedding (with the added
    /// conditioning), input convolution (plus <paramref name="addToInput"/>), the down levels and the middle block. Returns the skip features
    /// in the order they are produced, the middle block's output and the SiLU of the time embedding.
    /// </summary>
    private (List<Value> Skips, Value Middle, Value Time) Encoder(UNetHyperParameters hp, string prefix, Value input, Value timestep, Value? addToInput)
    {
        Value time;
        using (_b.Group(Scoped(prefix, "time_embedding")))
        {
            time = TimeEmbedding(timestep, hp, prefix);
            if (hp.ClassEmbeddingInput > 0)
            {
                time = _b.Add(time, AddedConditioning(hp, prefix));
            }

            time = _b.Silu(time);
        }

        var h = DiffusionBlocks.Conv(_b, input, prefix + "input_conv");
        if (addToInput is not null)
        {
            h = _b.Add(h, addToInput);
        }

        var skips = new List<Value> { h };
        int levels = hp.BlockChannels.Length;
        for (int level = 0; level < levels; level++)
        {
            using var _ = _b.Group(Scoped(prefix, $"down.{level}"));
            for (int i = 0; i < hp.LayersPerBlock; i++)
            {
                h = DiffusionBlocks.ResidualBlock(_b, h, time, $"{prefix}down.{level}.res.{i}", hp.NormGroups, hp.NormEps);
                if (hp.DownTransformerDepths[level] > 0)
                {
                    h = SpatialTransformer(h, $"{prefix}down.{level}.attn.{i}", hp.DownTransformerDepths[level], hp.HeadCounts[level], hp.NormGroups);
                }

                skips.Add(h);
            }

            if (level < levels - 1)
            {
                h = DiffusionBlocks.Conv(_b, h, $"{prefix}down.{level}.downsample", stride: 2);
                skips.Add(h);
            }
        }

        using (_b.Group(Scoped(prefix, "mid")))
        {
            h = DiffusionBlocks.ResidualBlock(_b, h, time, prefix + "mid.res.0", hp.NormGroups, hp.NormEps);
            if (hp.MidTransformerDepth > 0)
            {
                h = SpatialTransformer(h, prefix + "mid.attn.0", hp.MidTransformerDepth, hp.HeadCounts[^1], hp.NormGroups);
            }

            h = DiffusionBlocks.ResidualBlock(_b, h, time, prefix + "mid.res.1", hp.NormGroups, hp.NormEps);
        }

        return (skips, h, time);
    }

    /// <summary>
    /// The ControlNet's residuals for the UNet's skip features and middle block: its down half run on the (noisy) latents plus the control
    /// image's embedding, each feature through its zero convolution, scaled by the row's <see cref="DiffusionGraphs.ControlScale"/>.
    /// </summary>
    private (List<Value> Residuals, Value Middle) ControlResiduals(ControlNetHyperParameters control, Value latent, Value timestep, Dim batch,
        long latentHeight, long latentWidth)
    {
        string p = ControlNetHyperParameters.WeightPrefix;
        int levels = control.ConditionEmbeddingChannels.Length;
        long downscale = 1L << (levels - 1);
        var image = _b.Input(DiffusionGraphs.ControlImage, InputRole.Media, DataType.F32,
            new Shape(batch, control.ConditionChannels, latentHeight * downscale, latentWidth * downscale));
        var scale = _b.Reshape(_b.Input(DiffusionGraphs.ControlScale, InputRole.Other, DataType.F32, new Shape(batch)), 0, 1, 1, 1);
        Value embedding;
        using (_b.Group("control_embedding"))
        {
            embedding = _b.Silu(DiffusionBlocks.Conv(_b, image, p + "condition_embedding.input_conv"));
            for (int i = 0; i < 2 * (levels - 1); i++)
            {
                embedding = _b.Silu(DiffusionBlocks.Conv(_b, embedding, $"{p}condition_embedding.blocks.{i}", stride: i % 2 == 1 ? 2 : 1));
            }

            embedding = DiffusionBlocks.Conv(_b, embedding, p + "condition_embedding.output_conv");
        }

        var (features, middle, _) = Encoder(control.UNet, p, latent, timestep, embedding);
        var residuals = new List<Value>(features.Count);
        using (_b.Group("control_residuals"))
        {
            for (int i = 0; i < features.Count; i++)
            {
                residuals.Add(_b.Mul(DiffusionBlocks.Conv(_b, features[i], $"{p}zero_convs.{i}"), scale));
            }

            return (residuals, _b.Mul(DiffusionBlocks.Conv(_b, middle, p + "mid_zero_conv"), scale));
        }
    }

    /// <summary>A group or constant name: the UNet's own, the ControlNet's under <c>control.</c>.</summary>
    private static string Scoped(string prefix, string name) => prefix == UNetPrefix ? name : "control." + name;

    /// <summary>Sinusoidal embedding of the timesteps [N] → [N, C0], then the two-layer MLP → [N, 4·C0].</summary>
    private Value TimeEmbedding(Value timestep, UNetHyperParameters hp, string prefix)
    {
        var embedding = Sinusoids(_b.Reshape(timestep, 0, 1), hp.BlockChannels[0], 1, Scoped(prefix, "time_frequencies"), hp);
        var hidden = _b.Silu(_b.Linear(embedding, prefix + "time_embedding.linear1.weight", prefix + "time_embedding.linear1.bias"));
        return _b.Linear(hidden, prefix + "time_embedding.linear2.weight", prefix + "time_embedding.linear2.bias");
    }

    /// <summary>
    /// SDXL's added conditioning: the pooled text embedding [N, P] and the sinusoidal embeddings of the size conditions [N, 6] (the
    /// refiner: [N, 5], see <see cref="UNetHyperParameters.SizeConditions"/>; each <see cref="UNetHyperParameters.AdditionTimeDim"/> wide),
    /// concatenated and passed through the two-layer MLP <c>{prefix}class_embedding.linear1|linear2</c> → [N, 4·C0].
    /// </summary>
    private Value AddedConditioning(UNetHyperParameters hp, string prefix)
    {
        var pooled = _pooled ?? throw new NotSupportedException("The ControlNet has added conditioning (SDXL) but the UNet has none.");
        int conditions = (int)_sizes!.Shape[1].Fixed;
        var embedded = _b.Reshape(Sinusoids(_b.Reshape(_sizes, 0, conditions, 1), hp.AdditionTimeDim, 2, Scoped(prefix, "size_frequencies"), hp), 0, conditions * hp.AdditionTimeDim);
        var hidden = _b.Silu(_b.Linear(_b.Concat(1, pooled, embedded), prefix + "class_embedding.linear1.weight", prefix + "class_embedding.linear1.bias"));
        return _b.Linear(hidden, prefix + "class_embedding.linear2.weight", prefix + "class_embedding.linear2.bias");
    }

    /// <summary>
    /// Sinusoidal embedding of <paramref name="values"/> (last axis of size 1) into <paramref name="width"/> channels along
    /// <paramref name="axis"/>, the last one: cosines first when the model flips them, frequencies 10000^(-i / (half - shift)).
    /// </summary>
    private Value Sinusoids(Value values, int width, int axis, string name, UNetHyperParameters hp)
    {
        int half = width / 2;
        var frequencies = new float[half];
        for (int i = 0; i < half; i++)
        {
            frequencies[i] = (float)Math.Exp(-Math.Log(10000) * i / (half - hp.FreqShift));
        }

        var shape = axis == 1 ? new Shape(1, half) : new Shape(1, 1, half);
        var table = _b.Constant(name, DataType.F32, shape, MemoryMarshal.AsBytes(frequencies.AsSpan()).ToArray());
        var arguments = _b.Mul(values, table);
        var sin = _b.Unary("Sin", arguments);
        var cos = _b.Unary("Cos", arguments);
        return hp.FlipSinToCos ? _b.Concat(axis, cos, sin) : _b.Concat(axis, sin, cos);
    }

    /// <summary>
    /// Spatial transformer: GroupNorm, projection into token rows, <paramref name="depth"/> transformer blocks (self-attention,
    /// cross-attention to the text context, GEGLU feed-forward, each pre-normed and residual), projection back, plus the input.
    /// </summary>
    private Value SpatialTransformer(Value x, string prefix, int depth, int heads, int groups)
    {
        long height = x.Shape[2].Fixed;
        long width = x.Shape[3].Fixed;
        long pixels = height * width;
        var normed = DiffusionBlocks.GroupNorm(_b, x, prefix + ".norm", groups, TransformerNormEps);
        var rows = _b.Linear(DiffusionBlocks.ToRows(_b, normed), prefix + ".proj_in.weight", prefix + ".proj_in.bias");
        for (int block = 0; block < depth; block++)
        {
            string p = $"{prefix}.blocks.{block}";
            var selfInput = DiffusionBlocks.LayerNorm(_b, rows, p + ".norm1", LayerNormEps);
            rows = _b.Add(rows, DiffusionBlocks.Attention(_b, selfInput, selfInput, p + ".self", heads, pixels, pixels));
            var crossInput = DiffusionBlocks.LayerNorm(_b, rows, p + ".norm2", LayerNormEps);
            rows = _b.Add(rows, DiffusionBlocks.Attention(_b, crossInput, _context, p + ".cross", heads, pixels, _contextRows));
            rows = _b.Add(rows, FeedForward(DiffusionBlocks.LayerNorm(_b, rows, p + ".norm3", LayerNormEps), p + ".ff"));
        }

        rows = _b.Linear(rows, prefix + ".proj_out.weight", prefix + ".proj_out.bias");
        return _b.Add(DiffusionBlocks.FromRows(_b, rows, DiffusionBlocks.BatchOf(x), height, width), x);
    }

    /// <summary>GEGLU feed-forward: one projection to [value, gate], value · GELU(gate), output projection.</summary>
    private Value FeedForward(Value x, string prefix)
    {
        var projected = _b.Linear(x, prefix + ".proj.weight", prefix + ".proj.bias");
        long inner = projected.Shape[1].Fixed / 2;
        var value = _b.Slice(projected, 1, 0, inner);
        var gate = _b.Slice(projected, 1, inner, inner);
        var gated = _b.Mul(value, _b.Unary("Gelu", gate));
        return _b.Linear(gated, prefix + ".out.weight", prefix + ".out.bias");
    }
}
