using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Emits the graphs of VoxCPM2 around its two language models: the local feature encoder (latent patches to language model rows), the local diffusion transformer
/// (one velocity evaluation of a patch, conditional and unconditional together), and the projections between the language models (the scalar quantizer, the fusion of
/// the residual model's input, the diffusion condition and the stop logits). The local transformers are MiniCPM-4 layers (RMS norm, grouped-query attention with
/// LongRoPE over the row's position in its sequence, gated SiLU feed-forward) with bidirectional attention inside each packed sequence.
/// </summary>
internal static class VoxCpm2Graphs
{
    /// <summary>The local encoder graph.</summary>
    public const string EncodeGraph = "voxcpm2.encode";

    /// <summary>The diffusion transformer graph.</summary>
    public const string DitGraph = "voxcpm2.dit";

    /// <summary>The graph of the language models' junction (scalar quantizer and the residual model's input).</summary>
    public const string FuseGraph = "voxcpm2.fuse";

    /// <summary>The graph of the diffusion condition and the stop logits.</summary>
    public const string ConditionGraph = "voxcpm2.condition";

    /// <summary>The symbol of the encoder's patches.</summary>
    public const string PatchSymbol = "P";

    /// <summary>The rows of one diffusion sequence: the condition (the two language models' projections), the time, the previous patch and the noisy patch.</summary>
    public static int DitRows(VoxCpm2Parameters p) => ConditionRows + 1 + (2 * p.PatchSize);

    // The condition's rows: the text-semantic and the residual model's projections.
    private const int ConditionRows = 2;

    /// <summary>
    /// The local encoder: rows [T, feature] (one class row then the patch's frames, per patch; the class rows' values are ignored) with <c>class_mask</c> [T, 1],
    /// <c>positions</c> [T, 1] (the row's place in its patch) and <c>spans</c> [T, 2] (its patch's first row and row count) go in; the language model rows of the patches
    /// (<c>features</c> [P, hidden], the class rows picked by <c>class_rows</c> [P]) come out.
    /// </summary>
    /// <param name="p">The sizes.</param>
    /// <param name="weights">Finds a weight by name.</param>
    public static ComputeGraph BuildEncoder(VoxCpm2Parameters p, Func<string, WeightInfo?> weights)
    {
        var b = new GraphBuilder(EncodeGraph, weights);
        var T = Dim.Sym(WellKnown.TokenSymbol);
        var rows = b.Input("rows", InputRole.Other, DataType.F32, new Shape(T, p.FeatureSize));
        var mask = b.Input("class_mask", InputRole.Other, DataType.F32, new Shape(T, 1));
        var positions = b.Input("positions", InputRole.Positions, DataType.I32, new Shape(T, 1));
        var spans = b.Input("spans", InputRole.RowSpans, DataType.I32, new Shape(T, 2));
        var classRows = b.Input("class_rows", InputRole.Other, DataType.I32, new Shape(Dim.Sym(PatchSymbol)));
        var x = b.Linear(rows, "feat_encoder.in_proj.weight", "feat_encoder.in_proj.bias");
        x = b.Add(b.Mul(x, b.Scale(mask, -1f, 1f)), b.Mul(b.Weight("feat_encoder.special_token"), mask));
        x = Transformer(b, x, positions, spans, "feat_encoder.encoder.", p.Encoder, p);
        x = b.Gather(x, classRows);
        b.Output("features", b.Linear(x, "enc_to_lm_proj.weight", "enc_to_lm_proj.bias"));
        return b.Build();
    }

    /// <summary>
    /// The local diffusion transformer, conditional and unconditional sequences in one run: <c>mu</c> [2, hidden] (the condition, then zeros), <c>time</c> and
    /// <c>delta</c> [1, width] (the sinusoidal embeddings of the time and of the step), <c>cond</c> [patch, feature] (the previous patch) and <c>x</c> [patch, feature]
    /// (the current sample) go in; the velocities [2 · patch, feature] (conditional rows first) come out.
    /// </summary>
    /// <param name="p">The sizes.</param>
    /// <param name="weights">Finds a weight by name.</param>
    public static ComputeGraph BuildDit(VoxCpm2Parameters p, Func<string, WeightInfo?> weights)
    {
        const string W = "feat_decoder.estimator.";
        var b = new GraphBuilder(DitGraph, weights);
        int width = p.Dit.Hidden;
        const int muRows = ConditionRows;
        int sequence = DitRows(p);
        var mu = b.Input("mu", InputRole.Other, DataType.F32, new Shape(2, muRows * width));
        var time = b.Input("time", InputRole.Other, DataType.F32, new Shape(1, width));
        var delta = b.Input("delta", InputRole.Other, DataType.F32, new Shape(1, width));
        var cond = b.Input("cond", InputRole.Other, DataType.F32, new Shape(p.PatchSize, p.FeatureSize));
        var sample = b.Input("x", InputRole.Other, DataType.F32, new Shape(p.PatchSize, p.FeatureSize));
        var t = b.Add(Mlp(b, time, W + "time_mlp."), Mlp(b, delta, W + "delta_time_mlp."));
        var c = b.Linear(cond, W + "cond_proj.weight", W + "cond_proj.bias");
        var xs = b.Linear(sample, W + "in_proj.weight", W + "in_proj.bias");
        var muRowsValue = b.Reshape(mu, 2 * muRows, width);
        var seq = b.Concat(0, b.Slice(muRowsValue, 0, 0, muRows), t, c, xs, b.Slice(muRowsValue, 0, muRows, muRows), t, c, xs);
        var positions = new int[2 * sequence];
        var spans = new int[2 * sequence * 2];
        for (int r = 0; r < 2 * sequence; r++)
        {
            positions[r] = r % sequence;
            spans[(2 * r) + 0] = r / sequence * sequence;
            spans[(2 * r) + 1] = sequence;
        }

        var positionsValue = b.Constant("dit_positions", DataType.I32, new Shape(2 * sequence, 1), MemoryMarshal.AsBytes(positions.AsSpan()).ToArray());
        var spansValue = b.Constant("dit_spans", DataType.I32, new Shape(2 * sequence, 2), MemoryMarshal.AsBytes(spans.AsSpan()).ToArray());
        var hidden = Transformer(b, seq, positionsValue, spansValue, W + "decoder.", p.Dit, p);
        int skip = sequence - p.PatchSize;
        var outputs = b.Concat(0, b.Slice(hidden, 0, skip, p.PatchSize), b.Slice(hidden, 0, sequence + skip, p.PatchSize));
        b.Output("velocity", b.Linear(outputs, W + "out_proj.weight", W + "out_proj.bias"));
        return b.Build();
    }

    /// <summary>
    /// The junction of the language models over <c>hidden</c> [T, hidden] (the text-semantic model's output rows), <c>features</c> [T, hidden] (the local encoder's rows,
    /// zero for text) and <c>audio_mask</c> [T, 1]: <c>lm</c> = the scalar quantizer's image of audio rows (text rows pass unchanged) and <c>residual_input</c> = the fusion
    /// projection of [lm, features].
    /// </summary>
    /// <param name="p">The sizes.</param>
    /// <param name="weights">Finds a weight by name.</param>
    public static ComputeGraph BuildFuse(VoxCpm2Parameters p, Func<string, WeightInfo?> weights)
    {
        var b = new GraphBuilder(FuseGraph, weights);
        var T = Dim.Sym(WellKnown.TokenSymbol);
        var hidden = b.Input("hidden", InputRole.Other, DataType.F32, new Shape(T, p.Hidden));
        var features = b.Input("features", InputRole.Other, DataType.F32, new Shape(T, p.Hidden));
        var mask = b.Input("audio_mask", InputRole.Other, DataType.F32, new Shape(T, 1));
        var latent = b.Unary("Tanh", b.Linear(hidden, "fsq_layer.in_proj.weight", "fsq_layer.in_proj.bias"));
        float scale = p.QuantizerScale;
        latent = b.Scale(b.Unary("Round", b.Scale(latent, scale)), 1f / scale);
        var quantized = b.Linear(latent, "fsq_layer.out_proj.weight", "fsq_layer.out_proj.bias");
        var lm = b.Add(b.Mul(quantized, mask), b.Mul(hidden, b.Scale(mask, -1f, 1f)));
        b.Output("lm", lm);
        b.Output("residual_input", b.Linear(b.Concat(1, lm, b.Mul(features, mask)), "fusion_concat_proj.weight", "fusion_concat_proj.bias"));
        return b.Build();
    }

    /// <summary>
    /// The diffusion condition and the stop logits of one step: <c>lm</c> and <c>residual</c> [1, hidden] (the two language models' states) go in, <c>mu</c> [1, 2 · width]
    /// (their projections to the diffusion transformer's width, side by side) and <c>stop</c> [1, 2] come out.
    /// </summary>
    /// <param name="p">The sizes.</param>
    /// <param name="weights">Finds a weight by name.</param>
    public static ComputeGraph BuildCondition(VoxCpm2Parameters p, Func<string, WeightInfo?> weights)
    {
        var b = new GraphBuilder(ConditionGraph, weights);
        var lm = b.Input("lm", InputRole.Other, DataType.F32, new Shape(1, p.Hidden));
        var residual = b.Input("residual", InputRole.Other, DataType.F32, new Shape(1, p.Hidden));
        b.Output("mu", b.Concat(1, b.Linear(lm, "lm_to_dit_proj.weight", "lm_to_dit_proj.bias"), b.Linear(residual, "res_to_dit_proj.weight", "res_to_dit_proj.bias")));
        b.Output("stop", b.Linear(b.Silu(b.Linear(lm, "stop_proj.weight", "stop_proj.bias")), "stop_head.weight"));
        return b.Build();
    }

    // linear_2(silu(linear_1(x))).
    private static Value Mlp(GraphBuilder b, Value x, string prefix)
        => b.Linear(b.Silu(b.Linear(x, prefix + "linear_1.weight", prefix + "linear_1.bias")), prefix + "linear_2.weight", prefix + "linear_2.bias");

    // MiniCPM-4 layers with bidirectional attention inside each row's span, the final norm at the end.
    private static Value Transformer(GraphBuilder b, Value x, Value positions, Value spans, string prefix, VoxCpm2TransformerParameters t, VoxCpm2Parameters p)
    {
        var rope = Attributes.From(("dims", (long)t.HeadDim), ("base", p.RopeBase), ("style", "neox"));
        var attention = Attributes.From(("scale", 1f / MathF.Sqrt(t.HeadDim)), ("causal", false));
        var factors = b.TryWeight("rope.freq_factors");
        for (int il = 0; il < t.Layers; il++)
        {
            using var _ = b.Group($"{prefix}layers.{il}");
            string n = $"{prefix}layers.{il}.";
            var h = b.RmsNorm(x, b.Weight(n + "input_layernorm.weight"), p.NormEps);
            var q = b.Reshape(b.MatMul(h, b.Weight(n + "self_attn.q_proj.weight")), 0, t.Heads, t.HeadDim);
            var k = b.Reshape(b.MatMul(h, b.Weight(n + "self_attn.k_proj.weight")), 0, t.KvHeads, t.HeadDim);
            var v = b.Reshape(b.MatMul(h, b.Weight(n + "self_attn.v_proj.weight")), 0, t.KvHeads, t.HeadDim);
            q = b.Emit1("Rope", factors is null ? [q, positions] : [q, positions, factors], rope);
            k = b.Emit1("Rope", factors is null ? [k, positions] : [k, positions, factors], rope);
            var attended = b.Emit1("Attention", [q, k, v, spans], attention);
            x = b.Add(x, b.MatMul(b.Reshape(attended, 0, t.Heads * t.HeadDim), b.Weight(n + "self_attn.o_proj.weight")));
            h = b.RmsNorm(x, b.Weight(n + "post_attention_layernorm.weight"), p.NormEps);
            var gated = b.Mul(b.Silu(b.MatMul(h, b.Weight(n + "mlp.gate_proj.weight"))), b.MatMul(h, b.Weight(n + "mlp.up_proj.weight")));
            x = b.Add(x, b.MatMul(gated, b.Weight(n + "mlp.down_proj.weight")));
        }

        return b.RmsNorm(x, b.Weight(prefix + "norm.weight"), p.NormEps);
    }
}
