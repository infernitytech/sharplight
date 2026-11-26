using Sharplight.Graph;
using Sharplight.ImageGeneration;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Builds the graphs of a tiny autoencoder (TAESD): <see cref="DiffusionGraphs.TinyVaeDecode"/> (model latents → image in [0, 1]) and
/// <see cref="DiffusionGraphs.TinyVaeEncode"/> (image in [0, 1] → model latents), with the inputs and outputs of the VAE graphs, for one
/// fixed size and a symbolic batch. The layers are numbered as diffusers' <c>DecoderTiny</c> / <c>EncoderTiny</c> lists them.
/// </summary>
internal static class TinyAutoencoderGraphBuilder
{
    private const float ClampRange = 3f;

    /// <summary>Builds the decoder graph.</summary>
    /// <param name="weights">Canonical weight lookup.</param>
    /// <param name="hp">The hyper-parameters.</param>
    /// <param name="latentHeight">Latent height.</param>
    /// <param name="latentWidth">Latent width.</param>
    public static ComputeGraph BuildDecoder(Func<string, WeightInfo?> weights, TinyAutoencoderHyperParameters hp, long latentHeight, long latentWidth)
    {
        var b = new GraphBuilder(DiffusionGraphs.TinyVaeDecode, weights);
        var batch = Dim.Sym(DiffusionGraphs.BatchSymbol);
        var latent = b.Input(DiffusionGraphs.Latent, InputRole.Media, DataType.F32, new Shape(batch, hp.LatentChannels, latentHeight, latentWidth));
        string p = TinyAutoencoderHyperParameters.WeightPrefix + "decoder.layers.";
        // Soft clamp of the latents to ±3: 3 · tanh(z / 3).
        var z = b.Scale(latent, 1f / hp.ScalingFactor, hp.ShiftFactor);
        var h = b.Scale(b.Unary("Tanh", b.Scale(z, 1f / ClampRange)), ClampRange);
        h = b.Unary("Relu", DiffusionBlocks.Conv(b, h, p + "0"));
        int layer = 2;
        for (int level = 0; level < hp.DecoderBlocks.Length; level++)
        {
            using var _ = b.Group($"tiny_decoder.{level}");
            for (int i = 0; i < hp.DecoderBlocks[level]; i++)
            {
                h = Block(b, h, p + layer++);
            }

            if (level < hp.DecoderBlocks.Length - 1)
            {
                long[] size = [h.Shape[2].Fixed * hp.UpsampleFactor, h.Shape[3].Fixed * hp.UpsampleFactor];
                h = b.Emit1("Resize", [h], Attributes.From(("size", size), ("mode", "nearest"), ("coord", "asymmetric"), ("nearest_mode", "floor")));
                layer++;
            }

            h = DiffusionBlocks.Conv(b, h, p + layer++);
        }

        b.Output(DiffusionGraphs.Image, b.Emit1("Clip", [h], Attributes.From(("min", 0f), ("max", 1f))));
        return b.Build();
    }

    /// <summary>Builds the encoder graph.</summary>
    /// <param name="weights">Canonical weight lookup.</param>
    /// <param name="hp">The hyper-parameters.</param>
    /// <param name="height">Image height.</param>
    /// <param name="width">Image width.</param>
    public static ComputeGraph BuildEncoder(Func<string, WeightInfo?> weights, TinyAutoencoderHyperParameters hp, long height, long width)
    {
        var b = new GraphBuilder(DiffusionGraphs.TinyVaeEncode, weights);
        var batch = Dim.Sym(DiffusionGraphs.BatchSymbol);
        var h = b.Input(DiffusionGraphs.Image, InputRole.Media, DataType.F32, new Shape(batch, 3, height, width));
        string p = TinyAutoencoderHyperParameters.WeightPrefix + "encoder.layers.";
        int layer = 0;
        for (int level = 0; level < hp.EncoderBlocks.Length; level++)
        {
            using var _ = b.Group($"tiny_encoder.{level}");
            h = DiffusionBlocks.Conv(b, h, p + layer++, stride: level == 0 ? 1 : 2);
            for (int i = 0; i < hp.EncoderBlocks[level]; i++)
            {
                h = Block(b, h, p + layer++);
            }
        }

        h = DiffusionBlocks.Conv(b, h, p + layer);
        // Model latents from the autoencoder's: (z - shift) · scale.
        b.Output(DiffusionGraphs.Latent, b.Scale(h, hp.ScalingFactor, -hp.ShiftFactor * hp.ScalingFactor));
        return b.Build();
    }

    /// <summary>ReLU residual block: three 3×3 convolutions with ReLUs between, plus the input (through the 1×1 skip when present), ReLU.</summary>
    private static Value Block(GraphBuilder b, Value x, string prefix)
    {
        var h = b.Unary("Relu", DiffusionBlocks.Conv(b, x, prefix + ".conv1"));
        h = b.Unary("Relu", DiffusionBlocks.Conv(b, h, prefix + ".conv2"));
        h = DiffusionBlocks.Conv(b, h, prefix + ".conv3");
        var skip = b.HasWeight(prefix + ".skip.weight") ? DiffusionBlocks.Conv(b, x, prefix + ".skip") : x;
        return b.Unary("Relu", b.Add(h, skip));
    }
}
