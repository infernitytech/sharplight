using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// CosyVoice 3's flow (<c>CausalMaskedDiffWithDiT</c>) as two graphs of generic ops. <see cref="EncodeGraph"/> turns speech tokens (the voice's
/// prompt tokens, then the generated ones) into the flow's mean <c>mu</c> at the mel rate: the token embedding, the pre-lookahead layer (a convolution
/// that sees three tokens ahead, leaky ReLU, a causal convolution, the residual) and the repetition of every row for the two mel frames of a token;
/// and projects the normalized x-vector to the speaker condition. <see cref="EstimatorGraph"/> is one evaluation of the DiT vector field: the
/// noisy mel, the prompt condition, <c>mu</c> and the speaker condition projected together plus a causal convolutional position embedding (two
/// grouped convolutions with Mish, padded on the left), the time embedding, the DiT blocks (shared with F5-TTS, <see cref="F5TransformerGraph"/>;
/// CosyVoice applies the rotary embedding to the projections before the head split, which rotates only the first head's channels) with full attention, the final adaptive norm and the projection to the mel bins. The host runs the Euler steps with classifier-free guidance.
/// </summary>
internal static class CosyVoiceFlowGraph
{
    /// <summary>The encoder graph's name.</summary>
    public const string EncodeGraph = "cosyvoice_flow_encode";

    /// <summary>The estimator graph's name.</summary>
    public const string EstimatorGraph = "cosyvoice_flow_estimator";

    /// <summary>The speech tokens [<see cref="WellKnown.TokenSymbol"/>] (encoder).</summary>
    public const string TokensInput = "tokens";

    /// <summary>The L2-normalized x-vector [1, 192] (encoder).</summary>
    public const string SpeakerInput = "speaker";

    /// <summary>The mean [2 · tokens, mels] (encoder output).</summary>
    public const string MuOutput = "mu";

    /// <summary>The speaker condition [1, mels] (encoder output).</summary>
    public const string SpeakerOutput = "speaker_condition";

    /// <summary>The current sample [frames, mels] (estimator).</summary>
    public const string SampleInput = "x";

    /// <summary>The mean [frames, mels] (estimator; zero for the unconditional evaluation).</summary>
    public const string MuInput = "mu";

    /// <summary>The prompt mel, zero after the prompt [frames, mels] (estimator; zero for the unconditional evaluation).</summary>
    public const string CondInput = "cond";

    /// <summary>The speaker condition [1, mels] (estimator; zero for the unconditional evaluation).</summary>
    public const string SpeakerConditionInput = "spks";

    /// <summary>The time step [1, 1] (estimator).</summary>
    public const string TimeInput = "t";

    /// <summary>The frame positions 0 … frames − 1 [frames, 1] (estimator).</summary>
    public const string PositionsInput = "positions";

    /// <summary>The velocity [frames, mels] (estimator output).</summary>
    public const string VelocityOutput = "velocity";

    /// <summary>The symbol of the mel frames (estimator).</summary>
    public const string FrameSymbol = "frames";

    private const string F = "flow.";
    private const string E = "flow.decoder.estimator.";

    /// <summary>Builds the encoder graph.</summary>
    /// <param name="weights">Looks up the weights.</param>
    /// <param name="lookahead">The pre-lookahead tokens (3).</param>
    /// <param name="tokenMelRatio">Mel frames per token (2).</param>
    public static ComputeGraph BuildEncode(Func<string, WeightInfo?> weights, int lookahead, int tokenMelRatio)
    {
        var b = new GraphBuilder(EncodeGraph, weights);
        var tokens = b.Input(TokensInput, InputRole.TokenIds, DataType.I32, new Shape(Dim.Sym(WellKnown.TokenSymbol)));
        var speaker = b.Input(SpeakerInput, InputRole.Other, DataType.F32, new Shape(1, 192));
        var rows = b.Gather(b.Weight(F + "input_embedding.weight"), tokens, "flow.token_rows");
        var h = b.Emit1("Conv1d", [rows, b.Weight(F + "pre_lookahead_layer.conv1.weight"), b.Weight(F + "pre_lookahead_layer.conv1.bias")],
            Attributes.From(("pad_left", 0L), ("pad_right", (long)lookahead)));
        h = KokoroBlocks.LeakyRelu(b, h, 0.01f);
        h = b.Emit1("Conv1d", [h, b.Weight(F + "pre_lookahead_layer.conv2.weight"), b.Weight(F + "pre_lookahead_layer.conv2.bias")],
            Attributes.From(("pad_left", 2L), ("pad_right", 0L)));
        h = b.Add(h, rows);
        var mu = b.Emit1("Interpolate1d", [h], Attributes.From(("mode", "nearest"), ("up", (long)tokenMelRatio), ("down", 1L)));
        b.Output(MuOutput, mu);
        b.Output(SpeakerOutput, b.Linear(speaker, F + "spk_embed_affine_layer.weight", F + "spk_embed_affine_layer.bias"));
        return b.Build();
    }

    /// <summary>Builds the estimator graph.</summary>
    /// <param name="weights">Looks up the weights.</param>
    /// <param name="depth">The DiT blocks (22).</param>
    /// <param name="heads">The attention heads (16).</param>
    /// <param name="mels">The mel bins (80).</param>
    public static ComputeGraph BuildEstimator(Func<string, WeightInfo?> weights, int depth, int heads, int mels)
    {
        var b = new GraphBuilder(EstimatorGraph, weights);
        var n = Dim.Sym(FrameSymbol);
        var x = b.Input(SampleInput, InputRole.Other, DataType.F32, new Shape(n, mels));
        var mu = b.Input(MuInput, InputRole.Other, DataType.F32, new Shape(n, mels));
        var cond = b.Input(CondInput, InputRole.Other, DataType.F32, new Shape(n, mels));
        var spks = b.Input(SpeakerConditionInput, InputRole.Other, DataType.F32, new Shape(1, mels));
        var t = b.Input(TimeInput, InputRole.Other, DataType.F32, new Shape(1, 1));
        var positions = b.Input(PositionsInput, InputRole.Positions, DataType.I32, new Shape(n, 1));
        int dim = (int)weights(E + "proj_out.weight")!.Shape[1].Fixed;

        var h = b.Linear(b.Concat(1, x, cond, mu, KokoroBlocks.StyleRows(b, spks, x)), E + "input_embed.proj.weight", E + "input_embed.proj.bias", "flow.input");
        h = b.Add(h, CausalConvPosition(b, h));
        var ones = b.Constant("flow.norm_ones", DataType.F32, new Shape(dim), KokoroBlocks.Bytes([.. Enumerable.Repeat(1f, dim)]));
        var zeros = b.Constant("flow.norm_zeros", DataType.F32, new Shape(dim), KokoroBlocks.Bytes(new float[dim]));
        Value Norm(Value v) => b.Emit1("LayerNorm", [v, ones, zeros], Attributes.From(("eps", 1e-6f)));
        int timeChannels = (int)weights(E + "time_embed.time_mlp.0.weight")!.Shape[1].Fixed;
        var timeAct = b.Silu(F5TransformerGraph.TimeEmbedding(b, E, timeChannels, t));
        for (int i = 0; i < depth; i++)
        {
            h = F5TransformerGraph.Block(b, dim, heads, h, timeAct, positions, $"{E}transformer_blocks.{i}.", Norm, ropeHeads: 1);
        }

        var final = b.Linear(timeAct, E + "norm_out.linear.weight", E + "norm_out.linear.bias");
        h = F5TransformerGraph.Modulate(b, Norm(h), b.Slice(final, 1, 0, dim), b.Slice(final, 1, dim, dim));
        b.Output(VelocityOutput, b.Linear(h, E + "proj_out.weight", E + "proj_out.bias", VelocityOutput));
        return b.Build();
    }

    // CausalConvPositionEmbedding: two grouped convolutions (kernel 31, 16 groups) padded on the left, each followed by Mish.
    private static Value CausalConvPosition(GraphBuilder b, Value h)
    {
        const string M = E + "input_embed.conv_pos_embed.";
        long kernel = b.Weight(M + "conv1.0.weight").Shape[2].Fixed;
        long channels = h.Shape[1].Fixed;
        long groups = channels / b.Weight(M + "conv1.0.weight").Shape[1].Fixed;
        var attributes = Attributes.From(("pad_left", kernel - 1), ("pad_right", 0L), ("groups", groups));
        var y = b.Unary("Mish", b.Emit1("Conv1d", [h, b.Weight(M + "conv1.0.weight"), b.Weight(M + "conv1.0.bias")], attributes));
        return b.Unary("Mish", b.Emit1("Conv1d", [y, b.Weight(M + "conv2.0.weight"), b.Weight(M + "conv2.0.bias")], attributes));
    }
}
