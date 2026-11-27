using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;
using Sharplight.Vision;

namespace Sharplight.Architectures;

/// <summary>
/// Emits the <see cref="WellKnown.VisionEncodeGraph"/> of a ViT-style vision tower described by the canonical
/// <see cref="VisionParameters"/> and <c>vision.*</c> weights (documented on <see cref="VisionParameters"/>):
/// patch embedding (a MatMul over the patchified input, optionally normed: GLM-4V) + learned position embedding (bilinear or bicubic taps) →
/// optional class row (EVA2-CLIP: appended after the patches; attention without rotary embedding does not depend on row order, and the output
/// layout leaves it out) → pre-norm or post-norm transformer blocks (LayerNorm or RMS norm, fused or separate QKV bidirectional multi-head attention with axial 2-D rotary
/// embedding, over the whole image or within windows (Qwen2.5-VL), plain or gated MLP) → optional trunk post-norm (PaddleOCR-VL, GLM-4V) →
/// patch merger (optional per-patch norm, m×m merge, optional linear downsample and linear/LayerNorm/GELU projection (GLM-4V), plain or
/// gated 2-layer MLP) producing embeddings of the language model's hidden size, optionally followed by an output stage (HunyuanVL: a learned
/// separator row after every row of merged tokens, a linear projection, learned begin / end rows and a final norm, composed through the
/// output-layout input). DeepStack towers
/// (Qwen3-VL) also merge the outputs of selected layers (m×m merge, LayerNorm, 2-layer MLP each) and append them to the embedding row.
/// Shared by every architecture that uses such a tower; it does not know which file the weights came from.
/// </summary>
internal static class VisionTowerGraph
{
    public static ComputeGraph Build(VisionParameters p, Func<string, WeightInfo?> weights)
    {
        var b = new GraphBuilder(WellKnown.VisionEncodeGraph, weights);
        var N = Dim.Sym(VisionParameters.PatchSymbol);
        int H = p.HiddenSize;
        int heads = p.HeadCount;
        int D = p.HeadDim;

        Value x;
        Value positions;
        Value? embIndex = null;
        Value embWeight;
        Value? windowSpans = null;
        Value? windowOrder = null;
        Value? layout = null;
        using (b.Group("input"))
        {
            var patches = b.Input(VisionParameters.PatchesInput, InputRole.Media, DataType.F32, new Shape(N, p.PatchDim));
            positions = b.Input(VisionParameters.PositionsInput, InputRole.Positions, DataType.I32, new Shape(N, 2));
            if (!p.DensePositionWeights)
            {
                embIndex = b.Input(VisionParameters.EmbeddingIndexInput, InputRole.Other, DataType.I32, new Shape(N, p.PositionTaps));
            }

            embWeight = b.Input(VisionParameters.EmbeddingWeightInput, InputRole.Other, DataType.F32, new Shape(N, p.PositionTaps));
            if (p.WindowSize > 0 || p.TileSize > 0)
            {
                // Tiled towers (Idefics3) attend within each tile: one window per tile, already in output order.
                windowSpans = b.Input(VisionParameters.WindowSpansInput, InputRole.RowSpans, DataType.I32, new Shape(N, 2));
            }

            if (p.WindowSize > 0)
            {
                windowOrder = b.Input(VisionParameters.WindowOrderInput, InputRole.Other, DataType.I32, new Shape(Dim.Sym(VisionParameters.MergedSymbol)));
            }

            if (p.HasLayout)
            {
                layout = b.Input(VisionParameters.LayoutInput, InputRole.Other, DataType.I32, new Shape(Dim.Sym(VisionParameters.OutputSymbol)));
            }

            x = b.Linear(patches, "vision.patch_embed.weight", "vision.patch_embed.bias", "patch_embed");
            if (b.HasWeight("vision.patch_norm.weight"))
            {
                x = Norm(b, p, x, "vision.patch_norm");
            }
        }

        if (p.PositionEmbeddingCount > 0)
        {
            using var _ = b.Group("position_embedding");
            var table = b.Weight("vision.position_embedding.weight");
            Value? sum = null;
            if (embIndex is null)
            {
                // Dense resampling weights [N, count]: the product with the table [count, hidden] (MatMul takes it as [hidden, count]).
                sum = b.MatMul(embWeight, b.Contiguous(b.Permute(table, 1, 0)), "pos_embd_resampled");
            }

            for (int k = 0; embIndex is not null && k < p.PositionTaps; k++)
            {
                var idx = b.Reshape(b.Slice(embIndex, 1, k, 1), 0);
                var term = b.Mul(b.Gather(table, idx), b.Slice(embWeight, 1, k, 1));
                sum = sum is null ? term : b.Add(sum, term);
            }
            x = b.Add(x, sum!, "inp_pos_embd");
        }

        if (p.ClassToken)
        {
            var classRow = b.Add(Row(b, "vision.class_embedding"), Row(b, "vision.class_position_embedding"), "class_row");
            x = b.Concat(0, x, classRow);
        }

        Value? cos = null;
        Value? sinSigned = null;
        if (p.RopeBase > 0)
        {
            (cos, sinSigned) = RotaryTables(b, p, positions);
        }

        var deepstack = new List<Value>();
        for (int il = 0; il < p.LayerCount; il++)
        {
            using var _ = b.Group($"vision.layers.{il}");
            bool windowed = windowSpans is not null && Array.IndexOf(p.FullAttentionLayers, il) < 0;
            x = EncoderLayer(b, p, x, il, cos, sinSigned, windowed ? windowSpans : null);
            if (Array.IndexOf(p.DeepstackLayers, il) >= 0)
            {
                string d = $"vision.deepstack.{il}.";
                var merged = Norm(b, p, Merge(b, x, p), d + "norm");
                var dsUp = b.Linear(merged, d + "up.weight", d + "up.bias");
                deepstack.Add(b.Linear(Activation(b, dsUp, p.MergerActivation), d + "down.weight", d + "down.bias", $"{d}out"));
            }
        }

        if (b.HasWeight("vision.post_norm.weight"))
        {
            x = Norm(b, p, x, "vision.post_norm");
        }

        if (p.MergerPool)
        {
            using var _ = b.Group("merger");
            b.Output(VisionParameters.EmbeddingsOutput, PooledOutput(b, p, x));
            return b.Build();
        }

        using (b.Group("merger"))
        {
            var h = Merge(b, b.HasWeight("vision.merger.norm.weight") ? MergerNorm(b, p, x, "vision.merger.norm") : x, p);
            if (b.HasWeight("vision.merger.merged_norm.weight"))
            {
                // LFM2-VL: a LayerNorm of the merged rows (pixel unshuffle) before the MLP.
                h = MergerNorm(b, p, h, "vision.merger.merged_norm");
            }

            if (!b.HasWeight("vision.merger.up.weight"))
            {
                // Idefics3: the merged rows (pixel shuffle) projected by one linear map.
                b.Output(VisionParameters.EmbeddingsOutput, b.Linear(h, "vision.merger.out.weight", "vision.merger.out.bias", "merger_out"));
                return b.Build();
            }

            if (b.HasWeight("vision.merger.downsample.weight"))
            {
                // A stride-m convolution over each m×m window (GLM-4V), i.e. a linear map of the merged row.
                h = b.Linear(h, "vision.merger.downsample.weight", "vision.merger.downsample.bias");
            }

            if (b.HasWeight("vision.merger.proj.weight"))
            {
                // GLM-4V: linear → LayerNorm → exact GELU before the MLP.
                var proj = b.Linear(h, "vision.merger.proj.weight", "vision.merger.proj.bias");
                var normed = b.Emit1("LayerNorm", [proj, b.Weight("vision.merger.proj_norm.weight"), b.Weight("vision.merger.proj_norm.bias")],
                    Attributes.From(("eps", p.MergerNormEps ?? p.NormEps)));
                h = b.Unary("Gelu", normed);
            }

            var up = b.Linear(h, "vision.merger.up.weight", "vision.merger.up.bias");
            var act = b.HasWeight("vision.merger.gate.weight")
                ? b.Mul(Activation(b, b.Linear(h, "vision.merger.gate.weight", "vision.merger.gate.bias"), p.MergerActivation), up)
                : Activation(b, up, p.MergerActivation);
            bool outputStage = layout is not null || b.HasWeight("vision.merger.out.weight");
            var down = b.Linear(act, "vision.merger.down.weight", "vision.merger.down.bias", outputStage ? "merger_down" : VisionParameters.EmbeddingsOutput);
            var rows = deepstack.Count == 0 ? down : b.Concat(1, [down, .. deepstack]);
            // Windowed towers run window by window; the output goes back to merge-window raster order.
            rows = windowOrder is null ? rows : b.Gather(rows, windowOrder);
            b.Output(VisionParameters.EmbeddingsOutput, outputStage ? OutputStage(b, p, rows, layout) : rows);
        }
        return b.Build();
    }

    /// <summary>
    /// Gemma 4's output: the m×m patches of every merge unit averaged and scaled by √hidden, standardized (<c>vision.std_bias</c>, <c>vision.std_scale</c>),
    /// RMS-normed without weight and projected (<c>vision.merger.out.weight</c>, no bias). Gemma 3's: the plain average, RMS-normed with
    /// <c>vision.merger.pool_norm.weight</c> and projected.
    /// </summary>
    private static Value PooledOutput(GraphBuilder b, VisionParameters p, Value x)
    {
        int H = p.HiddenSize;
        var units = Merge(b, x, p);
        Value sum = b.Slice(units, 1, 0, H);
        for (int k = 1; k < p.MergeUnit; k++)
        {
            sum = b.Add(sum, b.Slice(units, 1, (long)k * H, H));
        }

        float scale = p.PoolScale == VisionParameters.MeanPoolScale ? 1f / p.MergeUnit : MathF.Sqrt(H) / p.MergeUnit;
        var pooled = b.Scale(sum, scale, 0f, "pooled");
        if (p.Standardize)
        {
            pooled = b.Mul(b.Sub(pooled, b.Weight("vision.std_bias")), b.Weight("vision.std_scale"), "standardized");
        }

        var normed = b.RmsNorm(pooled, b.HasWeight("vision.merger.pool_norm.weight") ? b.Weight("vision.merger.pool_norm.weight") : null, p.MergerNormEps ?? p.NormEps);
        return b.Linear(normed, "vision.merger.out.weight", "vision.merger.out.bias", "merger_out");
    }

    /// <summary>
    /// The output stage after the merger MLP: the table [merged rows | class row | separator] is projected (<c>vision.merger.out</c>), the begin / end rows are
    /// appended, the layout picks the output rows from it and <c>vision.merger.out_norm</c> norms them. Every step is row-wise, so the rows are
    /// composed last.
    /// </summary>
    private static Value OutputStage(GraphBuilder b, VisionParameters p, Value merged, Value? layout)
    {
        var table = merged;
        if (p.RowSeparator)
        {
            table = b.Concat(0, table, Row(b, "vision.merger.row_separator"));
        }

        if (b.HasWeight("vision.merger.out.weight"))
        {
            table = b.Linear(table, "vision.merger.out.weight", "vision.merger.out.bias", "merger_out");
        }

        if (p.ImageBoundary)
        {
            table = b.Concat(0, table, Row(b, "vision.merger.begin"), Row(b, "vision.merger.end"));
        }

        var rows = layout is null ? table : b.Gather(table, layout);
        return b.HasWeight("vision.merger.out_norm.weight") ? MergerNorm(b, p, rows, "vision.merger.out_norm") : rows;
    }

    // A learned vector [width] as one row [1, width].
    private static Value Row(GraphBuilder b, string name)
    {
        var w = b.Weight(name);
        return b.Reshape(w, 1, w.Shape[0].Fixed);
    }

    private static Value MergerNorm(GraphBuilder b, VisionParameters p, Value x, string name)
        => Norm(b, p with { NormType = p.MergerNormType ?? p.NormType }, x, name, p.MergerNormEps);

    /// <summary>
    /// One transformer block of the tower (<c>vision.layers.{il}.*</c>), pre-norm or post-norm (<see cref="VisionParameters.NormPlacement"/>): bidirectional
    /// multi-head attention (fused or separate QKV, optional rotary tables, optionally restricted by <paramref name="mask"/>: i32 window spans or an f32
    /// additive mask, see the Attention op) and a plain or gated MLP. Post-norm blocks (EVA2-CLIP) norm the attention and MLP outputs before their
    /// residuals: x + norm(attn(x)), x + norm(mlp(x)). With <paramref name="kvHeads"/> (separate projections, pre-norm only) keys and values have
    /// fewer heads, each shared by a group of query heads.
    /// </summary>
    internal static Value EncoderLayer(GraphBuilder b, VisionParameters p, Value x, int il, Value? cos, Value? sinSigned, Value? mask, int kvHeads = 0)
    {
        int H = p.HiddenSize;
        int heads = p.HeadCount;
        int D = p.HeadDim;
        Value? windowSpans = mask;
        string w = $"vision.layers.{il}.";
        // Row count: the symbolic patch count (kept by 0) or, in static graphs, the fixed one.
        long n = x.Shape[0].IsSymbolic ? 0 : x.Shape[0].Fixed;
        bool postNorm = p.NormPlacement == "post";
        bool sandwich = p.NormPlacement == "sandwich";
        float scale = p.AttentionScale != 0f ? p.AttentionScale : 1f / MathF.Sqrt(D);
        var h = postNorm ? x : Norm(b, p, x, w + "input_norm");
        Value qk;
        Value v;
        if (kvHeads > 0 && kvHeads != heads)
        {
            var gq = b.Reshape(b.Linear(h, w + "attention.q.weight", w + "attention.q.bias"), n, heads, D);
            var gk = b.Reshape(b.Linear(h, w + "attention.k.weight", w + "attention.k.bias"), n, kvHeads, D);
            var gv = b.Reshape(b.Linear(h, w + "attention.v.weight", w + "attention.v.bias"), n, kvHeads, D);
            if (cos is not null)
            {
                gq = Rotate(b, gq, cos, sinSigned!, D);
                gk = Rotate(b, gk, cos, sinSigned!, D);
            }

            var ga = b.Emit1("Attention", mask is not null ? [gq, gk, gv, mask] : [gq, gk, gv], Attributes.From(("scale", scale), ("causal", false)), $"{w}attn");
            return Mlp(b, p, b.Add(x, b.Linear(b.Reshape(ga, n, H), w + "attention.o.weight", w + "attention.o.bias"), $"{w}attn_residual"), w);
        }

        if (b.HasWeight(w + "attention.qkv.weight"))
        {
            var qkv = b.Linear(h, w + "attention.qkv.weight", w + "attention.qkv.bias");          // [N, 3H]: q | k | v
            qk = b.Reshape(b.Slice(qkv, 1, 0, 2L * H), n, 2 * heads, D);                       // [N, 2·heads, D]
            v = b.Reshape(b.Slice(qkv, 1, 2L * H, H), n, heads, D);
        }
        else
        {
            var q0 = b.Reshape(b.Linear(h, w + "attention.q.weight", w + "attention.q.bias"), n, heads, D);
            var k0 = b.Reshape(b.Linear(h, w + "attention.k.weight", w + "attention.k.bias"), n, heads, D);
            qk = b.Concat(1, q0, k0);
            v = b.Reshape(b.Linear(h, w + "attention.v.weight", w + "attention.v.bias"), n, heads, D);
        }

        if (p.QkNorm)
        {
            // Per-head RMS norms of the queries and keys (Gemma 4), before the rotation.
            qk = b.Concat(1, b.RmsNorm(b.Slice(qk, 1, 0, heads), b.Weight(w + "attention.q_norm.weight"), p.NormEps),
                b.RmsNorm(b.Slice(qk, 1, heads, heads), b.Weight(w + "attention.k_norm.weight"), p.NormEps));
        }

        if (p.ValueNorm)
        {
            v = b.RmsNorm(v, null, p.NormEps);
        }

        if (cos is not null)
        {
            qk = Rotate(b, qk, cos, sinSigned!, D, p.RopeLayout == VisionParameters.SplitRopeLayout);
        }

        var q = b.Slice(qk, 1, 0, heads);
        var k = b.Slice(qk, 1, heads, heads);
        var attn = b.Emit1("Attention", windowSpans is not null ? [q, k, v, windowSpans] : [q, k, v], Attributes.From(("scale", scale), ("causal", false)), $"{w}attn");
        var o = b.Linear(b.Reshape(attn, n, H), w + "attention.o.weight", w + "attention.o.bias");
        if (sandwich)
        {
            o = Norm(b, p, o, w + "attention.post_norm");
        }

        x = b.Add(x, postNorm ? Norm(b, p, o, w + "input_norm") : o, $"{w}attn_residual");
        return Mlp(b, p, x, w);
    }

    private static Value Mlp(GraphBuilder b, VisionParameters p, Value x, string w)
    {
        bool postNorm = p.NormPlacement == "post";
        var h = postNorm ? x : Norm(b, p, x, w + "post_attention_norm");
        var up = b.Linear(h, w + "mlp.up.weight", w + "mlp.up.bias");
        var act = b.HasWeight(w + "mlp.gate.weight")
            ? b.Mul(Activation(b, b.Linear(h, w + "mlp.gate.weight", w + "mlp.gate.bias"), p.Activation), up)
            : Activation(b, up, p.Activation);
        var down = b.Linear(act, w + "mlp.down.weight", w + "mlp.down.bias");
        if (p.NormPlacement == "sandwich")
        {
            down = Norm(b, p, down, w + "mlp.post_norm");
        }

        return b.Add(x, postNorm ? Norm(b, p, down, w + "post_attention_norm") : down, $"{w}l_out");
    }

    // m×m neighbouring patches (consecutive in merge-window order) → one row of m²·hidden.
    private static Value Merge(GraphBuilder b, Value x, VisionParameters p) => p.MergeUnit == 1 ? x
        : b.Emit1("Reshape", [x], Attributes.From(("shape", new long[] { 0, (long)p.MergeUnit * p.HiddenSize }), ("outer_symbol", VisionParameters.MergedSymbol)));

    internal static Value Norm(GraphBuilder b, VisionParameters p, Value x, string name, float? eps = null) => p.NormType == "rms"
        ? b.RmsNorm(x, b.Weight(name + ".weight"), eps ?? p.NormEps)
        : b.Emit1("LayerNorm", [x, b.Weight(name + ".weight"), b.Weight(name + ".bias")], Attributes.From(("eps", eps ?? p.NormEps)));

    internal static Value Activation(GraphBuilder b, Value x, string kind) => kind switch
    {
        "gelu_tanh" => b.Unary("GeluTanh", x),
        "gelu" => b.Unary("Gelu", x),
        "silu" => b.Silu(x),
        "quick_gelu" => b.Mul(x, b.Unary("Sigmoid", b.Scale(x, 1.702f))),
        _ => throw new NotSupportedException($"Vision activation '{kind}'."),
    };

    /// <summary>
    /// Axial 2-D rotary tables for positions (row, col): per axis head_dim/4 frequencies base^(-2j/(head_dim/2)); the angles
    /// [row·f | col·f] are repeated to span the head (rotation pairs (i, i + head_dim/2)). Returns cos and the sign-folded
    /// sin (negated on the first half) as [N, 1, D], so that rotate(x) = x·cos + swap_halves(x)·sinSigned.
    /// </summary>
    private static (Value Cos, Value SinSigned) RotaryTables(GraphBuilder b, VisionParameters p, Value positions)
    {
        int D = p.HeadDim;
        int F = D / 4;
        bool split = p.RopeLayout == VisionParameters.SplitRopeLayout;
        var freqs = new float[F];
        for (int j = 0; j < F; j++)
        {
            freqs[j] = (float)Math.Pow(p.RopeBase, -2.0 * j / (D / 2));
        }

        var signs = new float[D];
        for (int i = 0; i < D; i++)
        {
            // Paired: the first half of the head takes the negated sine. Split: the first quarter of each half does.
            signs[i] = (split ? i % (D / 2) < F : i < D / 2) ? -1f : 1f;
        }

        var cf = b.Constant("rope_freqs", DataType.F32, new Shape(1, F), Bytes(freqs));
        var cs = b.Constant("rope_signs", DataType.F32, new Shape(1, 1, D), Bytes(signs));

        var pos = b.Emit1("Cast", [positions], Attributes.From(("type", DataType.F32.Name)));
        var row = b.Mul(b.Slice(pos, 1, 0, 1), cf);
        var col = b.Mul(b.Slice(pos, 1, 1, 1), cf);
        // Paired: [row | col | row | col]. Split (Gemma 4, x = column first): [col | col | row | row].
        Value half = split ? b.Concat(1, col, col) : b.Concat(1, row, col);
        var angle = b.Reshape(split ? b.Concat(1, half, b.Concat(1, row, row)) : b.Concat(1, half, half), 0, 1, D);
        var cos = b.Unary("Cos", angle, "rope_cos");
        var sin = b.Mul(b.Unary("Sin", angle), cs, "rope_sin");
        return (cos, sin);
    }

    private static Value Rotate(GraphBuilder b, Value x, Value cos, Value sinSigned, int D, bool split = false)
    {
        int q = D / 4;
        var swapped = split
            ? b.Concat(2, b.Slice(x, 2, q, q), b.Slice(x, 2, 0, q), b.Slice(x, 2, 3L * q, q), b.Slice(x, 2, 2L * q, q))
            : b.Concat(2, b.Slice(x, 2, D / 2, D / 2), b.Slice(x, 2, 0, D / 2));
        return b.Add(b.Mul(x, cos), b.Mul(swapped, sinSigned));
    }

    private static byte[] Bytes(float[] v)
    {
        var r = new byte[v.Length * 4];
        Buffer.BlockCopy(v, 0, r, 0, r.Length);
        return r;
    }
}
