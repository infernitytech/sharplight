using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// One evaluation of F5-TTS's flow-matching field (its diffusion transformer) as a graph of generic ops over the mel frames <c>[N, mels]</c>: the
/// character ids (already aligned to the frames, 0 as filler) embedded, given absolute sinusoid positions and refined by ConvNeXt V2 blocks with the
/// filler rows zeroed; the noisy mel, the masked reference mel and the text rows projected together plus a convolutional position embedding (two
/// grouped convolutions with Mish); the time step's sinusoid through the time MLP; DiT blocks with adaptive LayerNorm modulation (shift, scale and
/// gate from the time embedding), rotary self-attention (interleaved pairs) and a tanh-GELU feed-forward; the final adaptive norm and the projection
/// to the mel bins. The host runs the Euler steps and the classifier-free guidance (a second evaluation without reference audio and text).
/// </summary>
internal static class F5TransformerGraph
{
    /// <summary>The graph name.</summary>
    public const string GraphName = "f5_transformer";

    /// <summary>The current sample [N, mels].</summary>
    public const string SampleInput = "x";

    /// <summary>The reference mel, zero after the reference [N, mels].</summary>
    public const string CondInput = "cond";

    /// <summary>The character ids plus one, 0 for filler [N].</summary>
    public const string TextInput = "text";

    /// <summary>1 for the frames that hold a character, 0 for filler [N, 1] (the unconditional pass zeroes the ids but keeps this mask).</summary>
    public const string TextMaskInput = "text_mask";

    /// <summary>The time step [1, 1].</summary>
    public const string TimeInput = "t";

    /// <summary>The frame positions 0 … N − 1 [N, 1].</summary>
    public const string PositionsInput = "positions";

    /// <summary>The output velocity [N, mels].</summary>
    public const string VelocityOutput = "velocity";

    private const string P = "f5.";

    /// <summary>Builds the graph.</summary>
    /// <param name="p">The hyper-parameters.</param>
    /// <param name="weights">Looks up the weights.</param>
    public static ComputeGraph Build(F5Parameters p, Func<string, WeightInfo?> weights)
    {
        var b = new GraphBuilder(GraphName, weights);
        var N = Dim.Sym(WellKnown.TokenSymbol);
        var x = b.Input(SampleInput, InputRole.Other, DataType.F32, new Shape(N, p.MelBins));
        var cond = b.Input(CondInput, InputRole.Other, DataType.F32, new Shape(N, p.MelBins));
        var text = b.Input(TextInput, InputRole.TokenIds, DataType.I32, new Shape(N));
        var textMask = b.Input(TextMaskInput, InputRole.Other, DataType.F32, new Shape(N, 1));
        var t = b.Input(TimeInput, InputRole.Other, DataType.F32, new Shape(1, 1));
        var positions = b.Input(PositionsInput, InputRole.Positions, DataType.I32, new Shape(N, 1));

        var textRows = TextEmbedding(b, p, text, textMask, positions);
        var h = b.Linear(b.Concat(1, x, cond, textRows), P + "input_embed.proj.weight", P + "input_embed.proj.bias", "f5.input");
        h = b.Add(h, ConvPosition(b, p, h));
        var ones = b.Constant("f5.norm_ones", DataType.F32, new Shape(p.Dim), KokoroBlocks.Bytes([.. Enumerable.Repeat(1f, p.Dim)]));
        var zeros = b.Constant("f5.norm_zeros", DataType.F32, new Shape(p.Dim), KokoroBlocks.Bytes(new float[p.Dim]));
        Value Norm(Value v) => b.Emit1("LayerNorm", [v, ones, zeros], Attributes.From(("eps", 1e-6f)));
        var time = TimeEmbedding(b, P, p.TimeChannels, t);
        var timeAct = b.Silu(time);
        for (int i = 0; i < p.Depth; i++)
        {
            h = Block(b, p.Dim, p.Heads, h, timeAct, positions, $"{P}transformer_blocks.{i}.", Norm);
        }

        var final = b.Linear(timeAct, P + "norm_out.linear.weight", P + "norm_out.linear.bias");
        var scale = b.Slice(final, 1, 0, p.Dim);
        var shift = b.Slice(final, 1, p.Dim, p.Dim);
        h = Modulate(b, Norm(h), scale, shift);
        var velocity = b.Linear(h, P + "proj_out.weight", P + "proj_out.bias", VelocityOutput);
        b.Output(VelocityOutput, velocity);
        return b.Build();
    }

    private static Value TextEmbedding(GraphBuilder b, F5Parameters p, Value text, Value keep, Value positions)
    {
        var rows = b.Gather(b.Weight(P + "text_embed.text_embed.weight"), text, "f5.text_rows");
        int half = p.TextDim / 2;
        var frequencies = new float[half];
        for (int i = 0; i < half; i++)
        {
            frequencies[i] = (float)(1.0 / Math.Pow(10000.0, 2.0 * i / p.TextDim));
        }

        var table = b.Constant("f5.text_frequencies", DataType.F32, new Shape(1, half), KokoroBlocks.Bytes(frequencies));
        var angles = b.Mul(b.Emit1("Cast", [positions], Attributes.From(("type", "f32"))), table);
        rows = b.Mul(b.Add(rows, b.Concat(1, b.Unary("Cos", angles), b.Unary("Sin", angles))), keep);
        for (int i = 0; i < p.TextBlocks; i++)
        {
            string m = $"{P}text_embed.text_blocks.{i}.";
            var y = b.Emit1("Conv1d", [rows, b.Weight(m + "dwconv.weight"), b.Weight(m + "dwconv.bias")],
                Attributes.From(("pad_left", 3L), ("pad_right", 3L), ("groups", (long)p.TextDim)));
            y = b.Emit1("LayerNorm", [y, b.Weight(m + "norm.weight"), b.Weight(m + "norm.bias")], Attributes.From(("eps", 1e-6f)));
            y = b.Unary("Gelu", b.Linear(y, m + "pwconv1.weight", m + "pwconv1.bias"));
            y = GlobalResponseNorm(b, y, m + "grn.");
            y = b.Linear(y, m + "pwconv2.weight", m + "pwconv2.bias");
            rows = b.Mul(b.Add(rows, y), keep);
        }

        return rows;
    }

    // GRN: gamma · (x · Gx / mean(Gx)) + beta + x with Gx the L2 norm of each channel over the frames.
    private static Value GlobalResponseNorm(GraphBuilder b, Value x, string m)
    {
        var sum = Attributes.From(("mode", "sum"), ("axes", new long[] { 0 }), ("keepdims", true));
        var g = b.Unary("Sqrt", b.Emit1("Reduce", [b.Unary("Square", x)], sum));                                                // [1, C]
        var mean = b.Emit1("Reduce", [g], Attributes.From(("mode", "mean"), ("axes", new long[] { 1 }), ("keepdims", true)));   // [1, 1]
        var n = b.Mul(g, b.Unary("Recip", b.Scale(mean, 1f, 1e-6f)));
        long channels = x.Shape[1].Fixed;
        var gamma = b.Reshape(b.Weight(m + "gamma"), 1, channels);
        var beta = b.Reshape(b.Weight(m + "beta"), 1, channels);
        return b.Add(b.Add(b.Mul(gamma, b.Mul(x, n)), beta), x);
    }

    private static Value ConvPosition(GraphBuilder b, F5Parameters p, Value h)
    {
        const string M = P + "input_embed.conv_pos_embed.conv1d.";
        long pad = p.ConvPositionKernel / 2;
        var attributes = Attributes.From(("pad_left", pad), ("pad_right", pad), ("groups", (long)p.ConvPositionGroups));
        var y = b.Unary("Mish", b.Emit1("Conv1d", [h, b.Weight(M + "0.weight"), b.Weight(M + "0.bias")], attributes));
        return b.Unary("Mish", b.Emit1("Conv1d", [y, b.Weight(M + "2.weight"), b.Weight(M + "2.bias")], attributes));
    }

    /// <summary>The DiT time embedding: the sinusoid at scale 1000 (sines then cosines), linear, SiLU, linear: [1, dim].</summary>
    /// <param name="b">The graph builder.</param>
    /// <param name="prefix">The prefix of the estimator's weights (<c>time_embed.time_mlp.*</c> under it).</param>
    /// <param name="timeChannels">The sinusoid's channels.</param>
    /// <param name="t">The time step [1, 1].</param>
    internal static Value TimeEmbedding(GraphBuilder b, string prefix, int timeChannels, Value t)
    {
        int half = timeChannels / 2;
        double step = Math.Log(10000) / (half - 1);
        var frequencies = new float[half];
        for (int i = 0; i < half; i++)
        {
            frequencies[i] = (float)(1000.0 * Math.Exp(-step * i));
        }

        var angles = b.Mul(t, b.Constant(prefix + "time_frequencies", DataType.F32, new Shape(1, half), KokoroBlocks.Bytes(frequencies)));
        var embedding = b.Concat(1, b.Unary("Sin", angles), b.Unary("Cos", angles));
        var hidden = b.Silu(b.Linear(embedding, prefix + "time_embed.time_mlp.0.weight", prefix + "time_embed.time_mlp.0.bias"));
        return b.Linear(hidden, prefix + "time_embed.time_mlp.2.weight", prefix + "time_embed.time_mlp.2.bias");
    }

    /// <summary>
    /// One DiT block: adaptive LayerNorm modulation from the time embedding, rotary self-attention (interleaved pairs), the gated residual, the
    /// modulated tanh-GELU feed-forward. The norms are LayerNorms without affine parameters (unit weight, zero bias), eps 1e-6.
    /// </summary>
    /// <param name="b">The graph builder.</param>
    /// <param name="dim">The width.</param>
    /// <param name="heads">The attention heads.</param>
    /// <param name="x">The rows [N, dim].</param>
    /// <param name="timeAct">SiLU of the time embedding [1, dim].</param>
    /// <param name="positions">The row positions [N, 1].</param>
    /// <param name="m">The block's weight prefix.</param>
    /// <param name="norm">The affine-free LayerNorm.</param>
    /// <param name="ropeHeads">
    /// The heads that get the rotary embedding: all of them (the default), or fewer for code that applies it to the projection before the head split
    /// (only the first head's channels are rotated, as in CosyVoice's copy of the F5 attention).
    /// </param>
    /// <param name="spans">The attention spans [N, 2] (first key row and count per query row, block-causal chunk attention), or <see langword="null"/> for full attention.</param>
    internal static Value Block(GraphBuilder b, int dim, int heads, Value x, Value timeAct, Value positions, string m, Func<Value, Value> norm, int? ropeHeads = null, Value? spans = null)
    {
        using var _ = b.Group(m.TrimEnd('.'));
        var mod = b.Linear(timeAct, m + "attn_norm.linear.weight", m + "attn_norm.linear.bias");                                   // [1, 6 · dim]
        Value Chunk(int i) => b.Slice(mod, 1, (long)i * dim, dim);
        var normed = Modulate(b, norm(x), Chunk(1), Chunk(0));
        int headDim = dim / heads;
        var rope = Attributes.From(("dims", (long)headDim), ("base", 10000f), ("style", "gptj"));
        int rotated = ropeHeads ?? heads;
        Value Rotate(Value v)
        {
            if (rotated >= heads)
            {
                return b.Emit1("Rope", [v, positions], rope);
            }

            return b.Concat(1, b.Emit1("Rope", [b.Slice(v, 1, 0, rotated), positions], rope), b.Slice(v, 1, rotated, heads - rotated));
        }

        var q = Rotate(b.Reshape(b.Linear(normed, m + "attn.to_q.weight", m + "attn.to_q.bias"), 0, heads, headDim));
        var k = Rotate(b.Reshape(b.Linear(normed, m + "attn.to_k.weight", m + "attn.to_k.bias"), 0, heads, headDim));
        var v = b.Reshape(b.Linear(normed, m + "attn.to_v.weight", m + "attn.to_v.bias"), 0, heads, headDim);
        var attention = b.Emit1("Attention", spans is null ? [q, k, v] : [q, k, v, spans], Attributes.From(("scale", 1f / MathF.Sqrt(headDim)), ("causal", false)), m + "attn");
        var output = b.Linear(b.Reshape(attention, 0, dim), m + "attn.to_out.0.weight", m + "attn.to_out.0.bias");
        x = b.Add(x, b.Mul(Chunk(2), output));
        var ffIn = Modulate(b, norm(x), Chunk(4), Chunk(3));
        var ff = b.Linear(b.Unary("GeluTanh", b.Linear(ffIn, m + "ff.ff.0.0.weight", m + "ff.ff.0.0.bias")), m + "ff.ff.2.weight", m + "ff.ff.2.bias");
        return b.Add(x, b.Mul(Chunk(5), ff), m + "out");
    }

    /// <summary>norm(x) · (1 + scale) + shift.</summary>
    /// <param name="b">The graph builder.</param>
    /// <param name="normed">The normalized rows.</param>
    /// <param name="scale">The scale row.</param>
    /// <param name="shift">The shift row.</param>
    internal static Value Modulate(GraphBuilder b, Value normed, Value scale, Value shift) => b.Add(b.Mul(normed, b.Scale(scale, 1f, 1f)), shift);
}
