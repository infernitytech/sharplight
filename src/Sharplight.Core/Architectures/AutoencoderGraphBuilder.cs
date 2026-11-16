using Sharplight.Graph;
using Sharplight.ImageGeneration;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Builds the VAE graphs of a KL autoencoder (the latent-diffusion VAE): <see cref="DiffusionGraphs.VaeDecode"/> (model latents →
/// image in [0, 1]) and <see cref="DiffusionGraphs.VaeEncode"/> (image in [0, 1] → scaled mean latents), for one fixed image size and
/// a symbolic batch.
/// </summary>
internal static class AutoencoderGraphBuilder
{
    private const float Eps = 1e-6f;

    /// <summary>Builds the decoder graph.</summary>
    /// <param name="weights">Canonical weight lookup.</param>
    /// <param name="hp">The VAE hyper-parameters.</param>
    /// <param name="latentHeight">Latent height.</param>
    /// <param name="latentWidth">Latent width.</param>
    public static ComputeGraph BuildDecoder(Func<string, WeightInfo?> weights, AutoencoderHyperParameters hp, long latentHeight, long latentWidth)
    {
        var b = new GraphBuilder(DiffusionGraphs.VaeDecode, weights);
        var batch = Dim.Sym(DiffusionGraphs.BatchSymbol);
        var latent = b.Input(DiffusionGraphs.Latent, InputRole.Media, DataType.F32, new Shape(batch, hp.LatentChannels, latentHeight, latentWidth));
        var z = b.Scale(latent, 1f / hp.ScalingFactor, hp.ShiftFactor);
        if (hp.QuantConvolutions)
        {
            z = DiffusionBlocks.Conv(b, z, "vae.post_quant_conv");
        }

        var h = DiffusionBlocks.Conv(b, z, "vae.decoder.input_conv");
        using (b.Group("decoder.mid"))
        {
            h = Middle(b, h, "vae.decoder.mid", hp.NormGroups);
        }

        int levels = hp.BlockChannels.Length;
        for (int up = 0; up < levels; up++)
        {
            using var _ = b.Group($"decoder.up.{up}");
            for (int i = 0; i <= hp.LayersPerBlock; i++)
            {
                h = DiffusionBlocks.ResidualBlock(b, h, null, $"vae.decoder.up.{up}.res.{i}", hp.NormGroups, Eps);
            }

            if (up < levels - 1)
            {
                h = DiffusionBlocks.Upsample(b, h, $"vae.decoder.up.{up}.upsample");
            }
        }

        using (b.Group("decoder.output"))
        {
            h = b.Silu(DiffusionBlocks.GroupNorm(b, h, "vae.decoder.output_norm", hp.NormGroups, Eps));
            h = DiffusionBlocks.Conv(b, h, "vae.decoder.output_conv");
            h = b.Emit1("Clip", [b.Scale(h, 0.5f, 0.5f)], Attributes.From(("min", 0f), ("max", 1f)));
        }

        b.Output(DiffusionGraphs.Image, h);
        return b.Build();
    }

    /// <summary>Builds the encoder graph.</summary>
    /// <param name="weights">Canonical weight lookup.</param>
    /// <param name="hp">The VAE hyper-parameters.</param>
    /// <param name="height">Image height.</param>
    /// <param name="width">Image width.</param>
    public static ComputeGraph BuildEncoder(Func<string, WeightInfo?> weights, AutoencoderHyperParameters hp, long height, long width)
    {
        var b = new GraphBuilder(DiffusionGraphs.VaeEncode, weights);
        var batch = Dim.Sym(DiffusionGraphs.BatchSymbol);
        var image = b.Input(DiffusionGraphs.Image, InputRole.Media, DataType.F32, new Shape(batch, hp.ImageChannels, height, width));
        var h = DiffusionBlocks.Conv(b, b.Scale(image, 2f, -1f), "vae.encoder.input_conv");
        int levels = hp.BlockChannels.Length;
        for (int level = 0; level < levels; level++)
        {
            using var _ = b.Group($"encoder.down.{level}");
            for (int i = 0; i < hp.LayersPerBlock; i++)
            {
                h = DiffusionBlocks.ResidualBlock(b, h, null, $"vae.encoder.down.{level}.res.{i}", hp.NormGroups, Eps);
            }

            if (level < levels - 1)
            {
                // The encoder's downsampling pads only the bottom and right edges (asymmetric padding of the original VAE).
                h = DiffusionBlocks.Conv(b, h, $"vae.encoder.down.{level}.downsample", stride: 2, pads: [0, 0, 1, 1]);
            }
        }

        using (b.Group("encoder.mid"))
        {
            h = Middle(b, h, "vae.encoder.mid", hp.NormGroups);
        }

        using (b.Group("encoder.output"))
        {
            h = b.Silu(DiffusionBlocks.GroupNorm(b, h, "vae.encoder.output_norm", hp.NormGroups, Eps));
            h = DiffusionBlocks.Conv(b, h, "vae.encoder.output_conv");
            if (hp.QuantConvolutions)
            {
                h = DiffusionBlocks.Conv(b, h, "vae.quant_conv");
            }

            var mean = b.Contiguous(b.Slice(h, 1, 0, hp.LatentChannels));
            h = b.Scale(mean, hp.ScalingFactor, -hp.ShiftFactor * hp.ScalingFactor);
        }

        b.Output(DiffusionGraphs.Latent, h);
        return b.Build();
    }

    /// <summary>The middle block: residual block, single-head self-attention over all pixels, residual block.</summary>
    private static Value Middle(GraphBuilder b, Value h, string prefix, int groups)
    {
        h = DiffusionBlocks.ResidualBlock(b, h, null, prefix + ".res.0", groups, Eps);
        long height = h.Shape[2].Fixed;
        long width = h.Shape[3].Fixed;
        string attention = prefix + ".attn.0";
        var rows = DiffusionBlocks.ToRows(b, DiffusionBlocks.GroupNorm(b, h, attention + ".norm", groups, Eps));
        var attended = DiffusionBlocks.Attention(b, rows, rows, attention, heads: 1, height * width, height * width);
        h = b.Add(DiffusionBlocks.FromRows(b, attended, DiffusionBlocks.BatchOf(h), height, width), h);
        return DiffusionBlocks.ResidualBlock(b, h, null, prefix + ".res.1", groups, Eps);
    }
}
