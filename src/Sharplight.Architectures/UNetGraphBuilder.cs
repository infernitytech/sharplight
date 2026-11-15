using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.ImageGeneration;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Builds the denoiser graph (<see cref="DiffusionGraphs.Denoise"/>) of a latent-diffusion UNet with cross-attention (SD 1.x / 2.x
/// layout): sinusoidal timestep embedding, levels of residual blocks with spatial transformers, a middle block and the mirrored up
/// path with skip connections, for one fixed latent size and a symbolic batch.
/// </summary>
internal sealed class UNetGraphBuilder
{
    private const float TransformerNormEps = 1e-6f;
    private const float LayerNormEps = 1e-5f;

    private readonly GraphBuilder _b;
    private readonly UNetHyperParameters _hp;
    private readonly long _contextRows;
    private Value _context = null!;

    private UNetGraphBuilder(Func<string, WeightInfo?> weights, UNetHyperParameters hp, long contextRows)
    {
        _b = new GraphBuilder(DiffusionGraphs.Denoise, weights);
        _hp = hp;
        _contextRows = contextRows;
    }

    /// <summary>Builds the graph.</summary>
    /// <param name="weights">Canonical weight lookup.</param>
    /// <param name="hp">The UNet hyper-parameters.</param>
    /// <param name="latentHeight">Latent height.</param>
    /// <param name="latentWidth">Latent width.</param>
    /// <param name="contextRows">Tokens of the text conditioning per image.</param>
    public static ComputeGraph Build(Func<string, WeightInfo?> weights, UNetHyperParameters hp, long latentHeight, long latentWidth, long contextRows)
        => new UNetGraphBuilder(weights, hp, contextRows).Build(latentHeight, latentWidth);

    private ComputeGraph Build(long latentHeight, long latentWidth)
    {
        var batch = Dim.Sym(DiffusionGraphs.BatchSymbol);
        var latent = _b.Input(DiffusionGraphs.Latent, InputRole.Media, DataType.F32, new Shape(batch, _hp.InChannels, latentHeight, latentWidth));
        var timestep = _b.Input(DiffusionGraphs.Timestep, InputRole.Other, DataType.F32, new Shape(batch));
        var context = _b.Input(DiffusionGraphs.Context, InputRole.Other, DataType.F32, new Shape(batch, _contextRows, _hp.ContextDim));
        _context = _b.Emit1("Reshape", [context], Attributes.From(("shape", new[] { 0, (long)_hp.ContextDim }), ("outer_symbol", $"{_contextRows}*{batch}")));

        Value time;
        using (_b.Group("time_embedding"))
        {
            time = _b.Silu(TimeEmbedding(timestep));
        }

        var h = DiffusionBlocks.Conv(_b, latent, "unet.input_conv");
        var skips = new Stack<Value>();
        skips.Push(h);
        int levels = _hp.BlockChannels.Length;
        for (int level = 0; level < levels; level++)
        {
            using var _ = _b.Group($"down.{level}");
            for (int i = 0; i < _hp.LayersPerBlock; i++)
            {
                h = DiffusionBlocks.ResidualBlock(_b, h, time, $"unet.down.{level}.res.{i}", _hp.NormGroups, _hp.NormEps);
                if (_hp.DownTransformerDepths[level] > 0)
                {
                    h = SpatialTransformer(h, $"unet.down.{level}.attn.{i}", _hp.DownTransformerDepths[level], _hp.HeadCounts[level]);
                }

                skips.Push(h);
            }

            if (level < levels - 1)
            {
                h = DiffusionBlocks.Conv(_b, h, $"unet.down.{level}.downsample", stride: 2);
                skips.Push(h);
            }
        }

        using (_b.Group("mid"))
        {
            h = DiffusionBlocks.ResidualBlock(_b, h, time, "unet.mid.res.0", _hp.NormGroups, _hp.NormEps);
            if (_hp.MidTransformerDepth > 0)
            {
                h = SpatialTransformer(h, "unet.mid.attn.0", _hp.MidTransformerDepth, _hp.HeadCounts[^1]);
            }

            h = DiffusionBlocks.ResidualBlock(_b, h, time, "unet.mid.res.1", _hp.NormGroups, _hp.NormEps);
        }

        for (int up = 0; up < levels; up++)
        {
            using var _ = _b.Group($"up.{up}");
            int heads = _hp.HeadCounts[levels - 1 - up];
            for (int i = 0; i <= _hp.LayersPerBlock; i++)
            {
                h = _b.Concat(1, h, skips.Pop());
                h = DiffusionBlocks.ResidualBlock(_b, h, time, $"unet.up.{up}.res.{i}", _hp.NormGroups, _hp.NormEps);
                if (_hp.UpTransformerDepths[up] > 0)
                {
                    h = SpatialTransformer(h, $"unet.up.{up}.attn.{i}", _hp.UpTransformerDepths[up], heads);
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

    /// <summary>Sinusoidal embedding of the timesteps [N] → [N, C0], then the two-layer MLP → [N, 4·C0].</summary>
    private Value TimeEmbedding(Value timestep)
    {
        int width = _hp.BlockChannels[0];
        int half = width / 2;
        var frequencies = new float[half];
        for (int i = 0; i < half; i++)
        {
            frequencies[i] = (float)Math.Exp(-Math.Log(10000) * i / (half - _hp.FreqShift));
        }

        var table = _b.Constant("time_frequencies", DataType.F32, new Shape(1, half), MemoryMarshal.AsBytes(frequencies.AsSpan()).ToArray());
        var arguments = _b.Mul(_b.Reshape(timestep, 0, 1), table);
        var sin = _b.Unary("Sin", arguments);
        var cos = _b.Unary("Cos", arguments);
        var embedding = _hp.FlipSinToCos ? _b.Concat(1, cos, sin) : _b.Concat(1, sin, cos);
        var hidden = _b.Silu(_b.Linear(embedding, "unet.time_embedding.linear1.weight", "unet.time_embedding.linear1.bias"));
        return _b.Linear(hidden, "unet.time_embedding.linear2.weight", "unet.time_embedding.linear2.bias");
    }

    /// <summary>
    /// Spatial transformer: GroupNorm, projection into token rows, <paramref name="depth"/> transformer blocks (self-attention,
    /// cross-attention to the text context, GEGLU feed-forward, each pre-normed and residual), projection back, plus the input.
    /// </summary>
    private Value SpatialTransformer(Value x, string prefix, int depth, int heads)
    {
        long height = x.Shape[2].Fixed;
        long width = x.Shape[3].Fixed;
        long pixels = height * width;
        var normed = DiffusionBlocks.GroupNorm(_b, x, prefix + ".norm", _hp.NormGroups, TransformerNormEps);
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
