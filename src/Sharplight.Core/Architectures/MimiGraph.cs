using Sharplight.Audio;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Emits Mimi as graphs of generic ops over time-major signals with a symbolic length <c>T</c>: the causal SEANet encoder (zero padding), the encoder transformer (layer norm,
/// rotary attention over a sliding causal window, layer scale), the replicate-padded downsampling convolution, and the split residual vector quantizer (a semantic stage and an acoustic
/// stage, each projecting the latent to the code space and searching the nearest codebook entry per residual stage); the decoder mirrors it (the depthwise transposed upsampling
/// convolution, the decoder transformer, the SEANet decoder). The graphs take the row positions of the rotary embedding and the attention spans (one sequence: every row sees rows 0 to the row count) as inputs.
/// </summary>
internal static class MimiGraph
{
    private const string PositionsInput = AudioCodecGraphs.PositionsInput;
    private const string SpansInput = AudioCodecGraphs.SpansInput;

    /// <summary>Builds the encoder graph: <c>audio</c> [T, 1] to the per-codebook codes.</summary>
    /// <param name="p">The hyper-parameters.</param>
    /// <param name="weights">Finds a weight by name.</param>
    public static ComputeGraph BuildEncoder(MimiParameters p, Func<string, WeightInfo?> weights)
    {
        var b = new GraphBuilder(AudioCodecGraphs.EncodeGraph, weights);
        var x = b.Input(AudioCodecGraphs.AudioInput, InputRole.Media, DataType.F32, new Shape(Dim.Sym(WellKnown.TokenSymbol), 1));
        int layer = 0;
        x = Conv(b, x, $"encoder.layers.{layer++}", p.KernelSize);
        for (int i = p.Ratios.Length - 1; i >= 0; i--)
        {
            for (int j = 0; j < p.ResidualLayers; j++)
            {
                x = Residual(b, p, x, $"encoder.layers.{layer++}", (long)Math.Pow(p.DilationGrowth, j));
            }

            long ratio = p.Ratios[i];
            x = Conv(b, b.Unary("Elu", x), $"encoder.layers.{layer + 1}", (int)(2 * ratio), stride: ratio);
            layer += 2;
        }

        x = Conv(b, b.Unary("Elu", x), $"encoder.layers.{layer + 1}", p.LastKernelSize);
        var positions = b.Input(PositionsInput, InputRole.Positions, DataType.I32, new Shape(x.Shape[0], 1));
        var spans = b.Input(SpansInput, InputRole.RowSpans, DataType.I32, new Shape(x.Shape[0], 2));
        x = Transformer(b, p, x, "encoder_transformer", positions, spans);

        // The downsampling convolution pads on the left by replicating the first frame.
        long total = (2 * p.LatentStride) - p.LatentStride;
        var first = b.Slice(x, 0, 0, 1);
        var padded = b.Concat(0, Enumerable.Repeat(first, (int)total).Append(x).ToArray());
        var latent = b.Emit1("Conv1d", [padded, b.Weight("downsample.weight")], Attributes.From(("stride", p.LatentStride)));

        // Split quantizer: both stages see the same latent; their codes are the semantic codebooks followed by the acoustic ones.
        int semantic = p.SemanticCodebooks;
        var semanticResidual = Conv1x1(b, latent, "quantizer.semantic.input_proj.weight");
        var acousticResidual = Conv1x1(b, latent, "quantizer.acoustic.input_proj.weight");
        for (int q = 0; q < p.Codebooks; q++)
        {
            using var _ = b.Group($"quantizer.{q}");
            bool isSemantic = q < semantic;
            string embedName = isSemantic ? $"quantizer.semantic.layers.{q}.embed" : $"quantizer.acoustic.layers.{q - semantic}.embed";
            var residual = isSemantic ? semanticResidual : acousticResidual;
            var embed = b.Weight(embedName);
            var score = b.MatMul(residual, embed);
            var norm = b.Reshape(b.Scale(b.Unary("ReduceSum", b.Unary("Square", embed)), 0.5f), 1, p.CodebookSize);
            var index = b.Unary("ArgMax", b.Sub(score, norm));
            b.Output(AudioCodecGraphs.CodesName(q), index);
            if (q + 1 < p.Codebooks)
            {
                var quantized = b.Gather(embed, b.Reshape(index, 0));
                if (isSemantic)
                {
                    semanticResidual = b.Sub(semanticResidual, quantized);
                }
                else
                {
                    acousticResidual = b.Sub(acousticResidual, quantized);
                }
            }
        }

        return b.Build();
    }

    /// <summary>Builds the decoder graph: the codes of the codebooks (each masked in or out) to <c>audio</c> [T · samples per frame, 1].</summary>
    /// <param name="p">The hyper-parameters.</param>
    /// <param name="weights">Finds a weight by name.</param>
    public static ComputeGraph BuildDecoder(MimiParameters p, Func<string, WeightInfo?> weights)
    {
        var b = new GraphBuilder(AudioCodecGraphs.DecodeGraph, weights);
        var frames = Dim.Sym(WellKnown.TokenSymbol);
        var mask = b.Input(AudioCodecGraphs.MaskInput, InputRole.Other, DataType.F32, new Shape(p.Codebooks));
        Value? semantic = null;
        Value? acoustic = null;
        for (int q = 0; q < p.Codebooks; q++)
        {
            bool isSemantic = q < p.SemanticCodebooks;
            string embedName = isSemantic ? $"quantizer.semantic.layers.{q}.embed" : $"quantizer.acoustic.layers.{q - p.SemanticCodebooks}.embed";
            var codes = b.Input(AudioCodecGraphs.CodesName(q), InputRole.TokenIds, DataType.I32, new Shape(frames));
            var vectors = b.Mul(b.Gather(b.Weight(embedName), codes), b.Slice(mask, 0, q, 1));
            if (isSemantic)
            {
                semantic = semantic is null ? vectors : b.Add(semantic, vectors);
            }
            else
            {
                acoustic = acoustic is null ? vectors : b.Add(acoustic, vectors);
            }
        }

        var latent = b.Add(Conv1x1(b, semantic!, "quantizer.semantic.output_proj.weight"), Conv1x1(b, acoustic!, "quantizer.acoustic.output_proj.weight"));
        long stride = p.LatentStride;
        var x = b.Emit1("ConvTranspose1d", [latent, b.Weight("upsample.weight")],
            Attributes.From(("stride", stride), ("crop_left", 0L), ("crop_right", stride), ("groups", (long)p.Hidden)));
        var positions = b.Input(PositionsInput, InputRole.Positions, DataType.I32, new Shape(x.Shape[0], 1));
        var spans = b.Input(SpansInput, InputRole.RowSpans, DataType.I32, new Shape(x.Shape[0], 2));
        x = Transformer(b, p, x, "decoder_transformer", positions, spans);

        int layer = 0;
        x = Conv(b, x, $"decoder.layers.{layer++}", p.KernelSize);
        foreach (long ratio in p.Ratios)
        {
            x = b.Unary("Elu", x);
            x = b.Emit1("ConvTranspose1d", [x, b.Weight($"decoder.layers.{layer + 1}.weight"), b.Weight($"decoder.layers.{layer + 1}.bias")],
                Attributes.From(("stride", ratio), ("crop_left", 0L), ("crop_right", ratio)));
            layer += 2;
            for (int j = 0; j < p.ResidualLayers; j++)
            {
                x = Residual(b, p, x, $"decoder.layers.{layer++}", (long)Math.Pow(p.DilationGrowth, j));
            }
        }

        x = Conv(b, b.Unary("Elu", x), $"decoder.layers.{layer + 1}", p.LastKernelSize);
        b.Output(AudioCodecGraphs.AudioOutput, x);
        return b.Build();
    }

    private static Value Transformer(GraphBuilder b, MimiParameters p, Value x, string prefix, Value positions, Value spans)
    {
        var rope = Attributes.From(("dims", (long)p.HeadDim), ("base", p.RopeBase), ("style", "neox"));
        var attention = Attributes.From(("scale", 1f / MathF.Sqrt(p.HeadDim)), ("causal", true), ("window", (long)p.Window));
        for (int il = 0; il < p.Layers; il++)
        {
            using var _ = b.Group($"{prefix}.layers.{il}");
            string n = $"{prefix}.layers.{il}.";
            var h = TowerBlocks.LayerNorm(b, x, n + "input_layernorm", p.Eps);
            var q = b.Emit1("Rope", [b.Reshape(b.MatMul(h, b.Weight(n + "self_attn.q_proj.weight")), 0, p.Heads, p.HeadDim), positions], rope);
            var k = b.Emit1("Rope", [b.Reshape(b.MatMul(h, b.Weight(n + "self_attn.k_proj.weight")), 0, p.Heads, p.HeadDim), positions], rope);
            var v = b.Reshape(b.MatMul(h, b.Weight(n + "self_attn.v_proj.weight")), 0, p.Heads, p.HeadDim);
            var attended = b.Emit1("Attention", [q, k, v, spans], attention);
            var o = b.MatMul(b.Reshape(attended, 0, p.Heads * p.HeadDim), b.Weight(n + "self_attn.o_proj.weight"));
            x = b.Add(x, b.Mul(o, b.Weight(n + "self_attn_layer_scale.scale")));
            h = TowerBlocks.LayerNorm(b, x, n + "post_attention_layernorm", p.Eps);
            var mlp = b.MatMul(TowerBlocks.Activation(b, b.MatMul(h, b.Weight(n + "mlp.fc1.weight")), p.Activation), b.Weight(n + "mlp.fc2.weight"));
            x = b.Add(x, b.Mul(mlp, b.Weight(n + "mlp_layer_scale.scale")));
        }

        return x;
    }

    private static Value Residual(GraphBuilder b, MimiParameters p, Value x, string name, long dilation)
    {
        var y = Conv(b, b.Unary("Elu", x), name + ".block.1", p.ResidualKernelSize, dilation: dilation);
        y = Conv(b, b.Unary("Elu", y), name + ".block.3", 1);
        return b.Add(x, y);
    }

    // A pointwise (kernel 1) convolution without bias.
    private static Value Conv1x1(GraphBuilder b, Value x, string weight) => b.Emit1("Conv1d", [x, b.Weight(weight)]);

    // The causal convolution with zero padding: padding_total = (kernel - 1) · dilation + 1 - stride on the left, none on the right; the bias is optional.
    private static Value Conv(GraphBuilder b, Value x, string name, int kernel, long stride = 1, long dilation = 1)
    {
        long total = (((kernel - 1) * dilation) + 1) - stride;
        var attributes = Attributes.From(("stride", stride), ("dilation", dilation), ("pad_left", total), ("pad_right", 0L));
        return b.Emit1("Conv1d", [x, b.Weight(name + ".weight"), b.Weight(name + ".bias")], attributes);
    }
}
