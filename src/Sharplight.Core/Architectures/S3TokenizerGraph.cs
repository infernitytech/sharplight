using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// The S3 speech tokenizer v2 (CosyVoice 2's, used by Chatterbox) as a graph: the log-mel frames through two strided convolutions with GELU
/// (four frames per token), pre-norm transformer blocks whose attention uses rotary positions (rotate-half over the whole head) and adds an FSMN
/// memory (a depthwise convolution over the values plus the values), and the finite scalar quantizer's down projection. The host takes
/// tanh, scales by 0.999, rounds and reads the eight ternary digits as the token (<see cref="ChatterboxVoiceCloning"/>).
/// </summary>
internal static class S3TokenizerGraph
{
    /// <summary>The graph name.</summary>
    public const string GraphName = "s3_tokenizer";

    /// <summary>The log-mel frames [frames, melBins] (frames symbolic: <see cref="WellKnown.TokenSymbol"/>).</summary>
    public const string FeaturesInput = "features";

    /// <summary>The token positions [tokens, 1] (0, 1, ...).</summary>
    public const string PositionsInput = "positions";

    /// <summary>The quantizer's projection [tokens, 8].</summary>
    public const string ProjectionOutput = "projection";

    /// <summary>Builds the graph.</summary>
    /// <param name="weights">Looks up the weights.</param>
    /// <param name="prefix">The tokenizer's weight prefix (<c>s3gen.tokenizer.</c>).</param>
    /// <param name="melBins">The mel channels.</param>
    /// <param name="layers">The transformer blocks.</param>
    /// <param name="heads">The attention heads.</param>
    /// <param name="hidden">The width.</param>
    public static ComputeGraph Build(Func<string, WeightInfo?> weights, string prefix, int melBins, int layers, int heads, int hidden)
    {
        var b = new GraphBuilder(GraphName, weights);
        string e = prefix + "encoder.";
        var x = b.Input(FeaturesInput, InputRole.AudioFeatures, DataType.F32, new Shape(Dim.Sym(WellKnown.TokenSymbol), melBins));
        var stride = Attributes.From(("stride", 2L), ("pad_left", 1L), ("pad_right", 1L));
        x = b.Unary("Gelu", b.Emit1("Conv1d", [x, b.Weight(e + "conv1.weight"), b.Weight(e + "conv1.bias")], stride));
        x = b.Unary("Gelu", b.Emit1("Conv1d", [x, b.Weight(e + "conv2.weight"), b.Weight(e + "conv2.bias")], stride));
        var positions = b.Input(PositionsInput, InputRole.Positions, DataType.I32, new Shape(x.Shape[0], 1));
        int headSize = hidden / heads;
        var rope = Attributes.From(("dims", (long)headSize), ("base", 10000f), ("style", "neox"));
        var fsmn = Attributes.From(("pad_left", 15L), ("pad_right", 15L), ("groups", (long)hidden));
        for (int l = 0; l < layers; l++)
        {
            string n = $"{e}blocks.{l}.";
            var h = TowerBlocks.LayerNorm(b, x, n + "attn_ln", 1e-5f);
            var q = b.Emit1("Rope", [b.Reshape(b.Linear(h, n + "attn.query.weight", n + "attn.query.bias"), 0, heads, headSize), positions], rope);
            var k = b.Emit1("Rope", [b.Reshape(b.Linear(h, n + "attn.key.weight"), 0, heads, headSize), positions], rope);
            var v = b.Linear(h, n + "attn.value.weight", n + "attn.value.bias");
            var attention = b.Emit1("Attention", [q, k, b.Reshape(v, 0, heads, headSize)],
                Attributes.From(("scale", 1f / MathF.Sqrt(headSize)), ("causal", false)));
            var memory = b.Add(b.Emit1("Conv1d", [v, b.Weight(n + "attn.fsmn_block.weight")], fsmn), v);
            var attended = b.Add(b.Linear(b.Reshape(attention, 0, hidden), n + "attn.out.weight", n + "attn.out.bias"), memory);
            x = b.Add(x, attended);
            h = TowerBlocks.LayerNorm(b, x, n + "mlp_ln", 1e-5f);
            x = b.Add(x, b.Linear(b.Unary("Gelu", b.Linear(h, n + "mlp.0.weight", n + "mlp.0.bias")), n + "mlp.2.weight", n + "mlp.2.bias"));
        }

        b.Output(ProjectionOutput, b.Linear(x, prefix + "quantizer._codebook.project_down.weight", prefix + "quantizer._codebook.project_down.bias"));
        return b.Build();
    }
}
