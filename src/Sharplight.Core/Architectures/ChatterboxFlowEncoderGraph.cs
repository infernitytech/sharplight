using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// The token-to-condition half of Chatterbox's S3Gen flow (CosyVoice 2's <c>UpsampleConformerEncoder</c>) as a graph of generic ops: the speech
/// tokens (the voice's prompt tokens followed by the generated ones) are embedded, projected with a LayerNorm and scaled by √d, run through
/// the pre-lookahead convolutions, pre-norm conformer layers with Transformer-XL relative-position attention (<c>RelativeAttention</c>, sinusoids of
/// the row distances) and SiLU feed-forward blocks, a nearest ×2 upsampling with a causal convolution, a second embedding and layer stack
/// at the mel frame rate, the final LayerNorm and the projection to the mel bins: <c>mu</c> [2 · tokens, 80], the mean the flow matching starts from.
/// A second output is the speaker condition: the normalized voice embedding through the speaker projection.
/// </summary>
internal static class ChatterboxFlowEncoderGraph
{
    /// <summary>The graph name.</summary>
    public const string GraphName = "s3gen_encode";

    /// <summary>The speech tokens [<see cref="WellKnown.TokenSymbol"/>].</summary>
    public const string TokensInput = "tokens";

    /// <summary>The row distances of the token-rate layers, T − 1 down to −(T − 1).</summary>
    public const string DistancesInput = "distances";

    /// <summary>The row distances of the frame-rate layers, 2T − 1 down to −(2T − 1).</summary>
    public const string UpDistancesInput = "up_distances";

    /// <summary>The voice's speaker embedding [1, 192].</summary>
    public const string SpeakerInput = "speaker";

    /// <summary>The output <c>mu</c> [2T, mel bins].</summary>
    public const string MuOutput = "mu";

    /// <summary>The output speaker condition [1, mel bins].</summary>
    public const string SpeakerOutput = "speaker_condition";

    private const string P = "s3gen.flow.";
    private const int Hidden = 512;
    private const int Heads = 8;
    private const float LayerEps = 1e-12f;

    /// <summary>Builds the graph.</summary>
    /// <param name="weights">Looks up the weights.</param>
    /// <param name="layers">Layers at the token rate.</param>
    /// <param name="upLayers">Layers at the frame rate.</param>
    public static ComputeGraph Build(Func<string, WeightInfo?> weights, int layers, int upLayers)
    {
        var b = new GraphBuilder(GraphName, weights);
        var T = Dim.Sym(WellKnown.TokenSymbol);
        var tokens = b.Input(TokensInput, InputRole.TokenIds, DataType.I32, new Shape(T));
        var distances = b.Input(DistancesInput, InputRole.Positions, DataType.I32, new Shape(T + T - Dim.Of(1), 1));
        var up = T + T;
        var upDistances = b.Input(UpDistancesInput, InputRole.Other, DataType.I32, new Shape(up + up - Dim.Of(1), 1));
        var speaker = b.Input(SpeakerInput, InputRole.Other, DataType.F32, new Shape(1, 192));

        var x = b.Gather(b.Weight(P + "input_embedding.weight"), tokens, "s3gen.token_embedding");
        x = Embed(b, x, P + "encoder.embed.out");
        var positions = PositionEmbeddings(b, distances, "s3gen.positions");
        x = PreLookahead(b, x, P + "encoder.pre_lookahead_layer.");
        for (int i = 0; i < layers; i++)
        {
            x = Layer(b, x, positions, $"{P}encoder.encoders.{i}.");
        }

        x = b.Emit1("Interpolate1d", [x], Attributes.From(("mode", "nearest"), ("up", 2L), ("down", 1L)));
        x = Conv(b, x, P + "encoder.up_layer.conv", 4, 0);
        x = Embed(b, x, P + "encoder.up_embed.out");
        var upPositions = PositionEmbeddings(b, upDistances, "s3gen.up_positions");
        for (int i = 0; i < upLayers; i++)
        {
            x = Layer(b, x, upPositions, $"{P}encoder.up_encoders.{i}.");
        }

        x = Norm(b, x, P + "encoder.after_norm", 1e-5f);
        var mu = b.Linear(x, P + "encoder_proj.weight", P + "encoder_proj.bias", MuOutput);
        b.Output(MuOutput, mu);

        // F.normalize then the affine layer.
        var norm = b.Unary("Sqrt", b.Scale(b.Emit1("Reduce", [b.Unary("Square", speaker)], Attributes.From(("mode", "sum"), ("axes", new long[] { 1 }), ("keepdims", true))), 1f, 1e-24f));
        var normalized = b.Mul(speaker, b.Unary("Recip", norm));
        var speakerCondition = b.Linear(normalized, P + "spk_embed_affine_layer.weight", P + "spk_embed_affine_layer.bias", SpeakerOutput);
        b.Output(SpeakerOutput, speakerCondition);
        return b.Build();
    }

    // LinearNoSubsampling: linear, LayerNorm, then the relative positional encoding's x · √d.
    private static Value Embed(GraphBuilder b, Value x, string module)
    {
        x = b.Linear(x, module + ".0.weight", module + ".0.bias");
        x = Norm(b, x, module + ".1", 1e-5f);
        return b.Scale(x, MathF.Sqrt(Hidden));
    }

    private static Value Layer(GraphBuilder b, Value x, Value positions, string p)
    {
        using var _ = b.Group(p.TrimEnd('.'));
        int headDim = Hidden / Heads;
        var h = Norm(b, x, p + "norm_mha", LayerEps);
        var q = b.Reshape(b.Linear(h, p + "self_attn.linear_q.weight", p + "self_attn.linear_q.bias"), 0, Heads, headDim);
        var k = b.Reshape(b.Linear(h, p + "self_attn.linear_k.weight", p + "self_attn.linear_k.bias"), 0, Heads, headDim);
        var v = b.Reshape(b.Linear(h, p + "self_attn.linear_v.weight", p + "self_attn.linear_v.bias"), 0, Heads, headDim);
        var relative = b.Reshape(b.Linear(positions, p + "self_attn.linear_pos.weight"), 0, Heads, headDim);
        var content = b.Add(q, b.Weight(p + "self_attn.pos_bias_u"));
        var position = b.Add(q, b.Weight(p + "self_attn.pos_bias_v"));
        var attention = b.Emit1("RelativeAttention", [content, position, k, v, relative], Attributes.From(("scale", 1f / MathF.Sqrt(headDim))), p + "attn");
        x = b.Add(x, b.Linear(b.Reshape(attention, 0, Hidden), p + "self_attn.linear_out.weight", p + "self_attn.linear_out.bias"));
        h = Norm(b, x, p + "norm_ff", LayerEps);
        var ff = b.Linear(b.Silu(b.Linear(h, p + "feed_forward.w_1.weight", p + "feed_forward.w_1.bias")), p + "feed_forward.w_2.weight", p + "feed_forward.w_2.bias");
        return b.Add(x, ff, p + "out");
    }

    // Right-padded convolution (the lookahead), leaky ReLU, left-padded convolution, plus the input.
    private static Value PreLookahead(GraphBuilder b, Value x, string p)
    {
        var h = Conv(b, x, p + "conv1", 0, 3);
        h = b.Emit1("LeakyRelu", [h], Attributes.From(("alpha", 0.01f)));
        h = Conv(b, h, p + "conv2", 2, 0);
        return b.Add(h, x);
    }

    private static Value Conv(GraphBuilder b, Value x, string module, long padLeft, long padRight)
        => b.Emit1("Conv1d", [x, b.Weight(module + ".weight"), b.Weight(module + ".bias")], Attributes.From(("pad_left", padLeft), ("pad_right", padRight)));

    private static Value Norm(GraphBuilder b, Value x, string module, float eps)
        => b.Emit1("LayerNorm", [x, b.Weight(module + ".weight"), b.Weight(module + ".bias")], Attributes.From(("eps", eps)));

    // ESPnet's relative positional encoding: sin and cos of distance · 10000^(−2i/d), interleaved: [P, d].
    private static Value PositionEmbeddings(GraphBuilder b, Value distances, string name)
    {
        int half = Hidden / 2;
        var frequencies = new float[half];
        for (int i = 0; i < half; i++)
        {
            frequencies[i] = MathF.Exp(2 * i * -(MathF.Log(10000f) / Hidden));
        }

        var table = b.Constant(name + ".frequencies", DataType.F32, new Shape(1, half), KokoroBlocks.Bytes(frequencies));
        var angles = b.Mul(b.Emit1("Cast", [distances], Attributes.From(("type", "f32"))), table);
        var sine = b.Reshape(b.Unary("Sin", angles), 0, half, 1);
        var cosine = b.Reshape(b.Unary("Cos", angles), 0, half, 1);
        return b.Reshape(b.Concat(2, sine, cosine), 0, Hidden);
    }
}
