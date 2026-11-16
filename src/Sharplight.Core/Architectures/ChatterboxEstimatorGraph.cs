using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// One evaluation of S3Gen's flow-matching vector field (CosyVoice 2's causal <c>ConditionalDecoder</c>) as a graph of generic ops, over time-major
/// signals <c>[M, C]</c>: the sinusoid of the time step through the time MLP; the channels of the noisy mel, <c>mu</c>, the speaker condition (repeated
/// over the frames) and the prompt mel stacked to 320; one down block, the middle blocks and one up block (with the down block's output as skip),
/// each a causal residual block (causal convolution, LayerNorm over channels and Mish twice, the time embedding added between them, a 1×1 residual
/// projection) followed by pre-norm transformer blocks (bias-free attention, GELU feed-forward) and a causal convolution; the final causal block and
/// the 1×1 projection to the mel bins. The host runs the Euler steps and the classifier-free guidance (a second evaluation with zero conditions).
/// </summary>
internal static class ChatterboxEstimatorGraph
{
    /// <summary>The graph name.</summary>
    public const string GraphName = "s3gen_estimate";

    /// <summary>The current sample <c>x</c> [M, 80].</summary>
    public const string SampleInput = "x";

    /// <summary>The encoder's mean <c>mu</c> [M, 80].</summary>
    public const string MuInput = "mu";

    /// <summary>The speaker condition [1, 80].</summary>
    public const string SpeakerInput = "speaker";

    /// <summary>The prompt mel, zero after the prompt [M, 80].</summary>
    public const string CondInput = "cond";

    /// <summary>The time step [1].</summary>
    public const string TimeInput = "t";

    /// <summary>The output velocity [M, 80].</summary>
    public const string VelocityOutput = "velocity";

    /// <summary>The frame symbol.</summary>
    public const string FrameSymbol = "M";

    private const string P = "s3gen.flow.decoder.estimator.";
    private const int Heads = 8;
    private const int HeadDim = 64;
    private const int TimeChannels = 320;

    /// <summary>Builds the graph.</summary>
    /// <param name="weights">Looks up the weights.</param>
    /// <param name="midBlocks">The number of middle blocks.</param>
    /// <param name="transformers">Transformer blocks per block.</param>
    public static ComputeGraph Build(Func<string, WeightInfo?> weights, int midBlocks, int transformers)
    {
        var b = new GraphBuilder(GraphName, weights);
        var M = Dim.Sym(FrameSymbol);
        var x = b.Input(SampleInput, InputRole.Other, DataType.F32, new Shape(M, 80));
        var mu = b.Input(MuInput, InputRole.Other, DataType.F32, new Shape(M, 80));
        var speaker = b.Input(SpeakerInput, InputRole.Other, DataType.F32, new Shape(1, 80));
        var cond = b.Input(CondInput, InputRole.Other, DataType.F32, new Shape(M, 80));
        var t = b.Input(TimeInput, InputRole.Other, DataType.F32, new Shape(1, 1));

        var time = TimeEmbedding(b, t);
        var speakerRows = b.Add(b.Scale(mu, 0f), speaker);
        var h = b.Concat(1, x, mu, speakerRows, cond);

        h = ResnetBlock(b, h, time, P + "down_blocks.0.0.");
        h = Transformers(b, h, P + "down_blocks.0.1.", transformers);
        var skip = h;
        h = CausalConv(b, h, P + "down_blocks.0.2");
        for (int i = 0; i < midBlocks; i++)
        {
            h = ResnetBlock(b, h, time, $"{P}mid_blocks.{i}.0.");
            h = Transformers(b, h, $"{P}mid_blocks.{i}.1.", transformers);
        }

        h = b.Concat(1, h, skip);
        h = ResnetBlock(b, h, time, P + "up_blocks.0.0.");
        h = Transformers(b, h, P + "up_blocks.0.1.", transformers);
        h = CausalConv(b, h, P + "up_blocks.0.2");
        h = CausalBlock(b, h, P + "final_block.block");
        var velocity = Conv(b, h, P + "final_proj", 0, VelocityOutput);
        b.Output(VelocityOutput, velocity);
        return b.Build();
    }

    // SinusoidalPosEmb(320) at scale 1000 (sines then cosines), then linear, SiLU, linear: [1, 1024].
    private static Value TimeEmbedding(GraphBuilder b, Value t)
    {
        int half = TimeChannels / 2;
        double step = Math.Log(10000) / (half - 1);
        var frequencies = new float[half];
        for (int i = 0; i < half; i++)
        {
            frequencies[i] = (float)(1000.0 * Math.Exp(-step * i));
        }

        var table = b.Constant("s3gen.time_frequencies", DataType.F32, new Shape(1, half), KokoroBlocks.Bytes(frequencies));
        var angles = b.Mul(t, table);
        var embedding = b.Concat(1, b.Unary("Sin", angles), b.Unary("Cos", angles));
        var hidden = b.Silu(b.Linear(embedding, P + "time_mlp.linear_1.weight", P + "time_mlp.linear_1.bias"));
        return b.Linear(hidden, P + "time_mlp.linear_2.weight", P + "time_mlp.linear_2.bias");
    }

    private static Value ResnetBlock(GraphBuilder b, Value x, Value time, string p)
    {
        var h = CausalBlock(b, x, p + "block1.block");
        h = b.Add(h, b.Linear(b.Unary("Mish", time), p + "mlp.1.weight", p + "mlp.1.bias"));
        h = CausalBlock(b, h, p + "block2.block");
        return b.Add(h, Conv(b, x, p + "res_conv", 0));
    }

    // Causal convolution (kernel 3, two zero frames in front), LayerNorm over channels, Mish.
    private static Value CausalBlock(GraphBuilder b, Value x, string p)
    {
        var h = Conv(b, x, p + ".0", 2);
        h = b.Emit1("LayerNorm", [h, b.Weight(p + ".2.weight"), b.Weight(p + ".2.bias")], Attributes.From(("eps", 1e-5f)));
        return b.Unary("Mish", h);
    }

    private static Value CausalConv(GraphBuilder b, Value x, string module) => Conv(b, x, module, 2);

    private static Value Conv(GraphBuilder b, Value x, string module, long padLeft, string? name = null)
        => b.Emit1("Conv1d", [x, b.Weight(module + ".weight"), b.Weight(module + ".bias")], Attributes.From(("pad_left", padLeft), ("pad_right", 0L)), name);

    private static Value Transformers(GraphBuilder b, Value x, string p, int count)
    {
        long width = (long)Heads * HeadDim;
        for (int i = 0; i < count; i++)
        {
            string l = $"{p}{i}.";
            using var _ = b.Group(l.TrimEnd('.'));
            var h = Norm(b, x, l + "norm1");
            var q = b.Reshape(b.Linear(h, l + "attn1.to_q.weight"), 0, Heads, HeadDim);
            var k = b.Reshape(b.Linear(h, l + "attn1.to_k.weight"), 0, Heads, HeadDim);
            var v = b.Reshape(b.Linear(h, l + "attn1.to_v.weight"), 0, Heads, HeadDim);
            var attention = b.Emit1("Attention", [q, k, v], Attributes.From(("scale", 1f / MathF.Sqrt(HeadDim)), ("causal", false)), l + "attn");
            x = b.Add(x, b.Linear(b.Reshape(attention, 0, width), l + "attn1.to_out.0.weight", l + "attn1.to_out.0.bias"));
            h = Norm(b, x, l + "norm3");
            var ff = b.Linear(b.Unary("Gelu", b.Linear(h, l + "ff.net.0.proj.weight", l + "ff.net.0.proj.bias")), l + "ff.net.2.weight", l + "ff.net.2.bias");
            x = b.Add(x, ff, l + "out");
        }

        return x;
    }

    private static Value Norm(GraphBuilder b, Value x, string module)
        => b.Emit1("LayerNorm", [x, b.Weight(module + ".weight"), b.Weight(module + ".bias")], Attributes.From(("eps", 1e-5f)));
}
