using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;
using Sharplight.Vision;

namespace Sharplight.Architectures;

/// <summary>
/// Emits the <see cref="WellKnown.VisionEncodeGraph"/> of a ViT-style vision tower described by the canonical
/// <see cref="VisionParameters"/> and <c>vision.*</c> weights (documented on <see cref="VisionParameters"/>):
/// patch embedding (a MatMul over the patchified input) + learned position embedding (bilinear taps) → pre-norm
/// transformer blocks (LayerNorm or RMS norm, fused or separate QKV bidirectional multi-head attention with axial 2-D rotary embedding, over
/// the whole image or within windows (Qwen2.5-VL), plain or gated MLP) → patch merger (norm, m×m merge, 2-layer MLP) producing embeddings of the language model's hidden size. DeepStack towers
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
        Value embIndex, embWeight;
        Value? windowSpans = null;
        Value? windowOrder = null;
        using (b.Group("input"))
        {
            var patches = b.Input(VisionParameters.PatchesInput, InputRole.Media, DataType.F32, new Shape(N, p.PatchDim));
            positions = b.Input(VisionParameters.PositionsInput, InputRole.Positions, DataType.I32, new Shape(N, 2));
            embIndex = b.Input(VisionParameters.EmbeddingIndexInput, InputRole.Other, DataType.I32, new Shape(N, 4));
            embWeight = b.Input(VisionParameters.EmbeddingWeightInput, InputRole.Other, DataType.F32, new Shape(N, 4));
            if (p.WindowSize > 0)
            {
                windowSpans = b.Input(VisionParameters.WindowSpansInput, InputRole.RowSpans, DataType.I32, new Shape(N, 2));
                windowOrder = b.Input(VisionParameters.WindowOrderInput, InputRole.Other, DataType.I32, new Shape(Dim.Sym(VisionParameters.MergedSymbol)));
            }

            x = b.Linear(patches, "vision.patch_embed.weight", "vision.patch_embed.bias", "patch_embed");
        }

        if (p.PositionEmbeddingCount > 0)
        {
            using var _ = b.Group("position_embedding");
            var table = b.Weight("vision.position_embedding.weight");
            Value? sum = null;
            for (int k = 0; k < 4; k++)
            {
                var idx = b.Reshape(b.Slice(embIndex, 1, k, 1), 0);
                var term = b.Mul(b.Gather(table, idx), b.Slice(embWeight, 1, k, 1));
                sum = sum is null ? term : b.Add(sum, term);
            }
            x = b.Add(x, sum!, "inp_pos_embd");
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
            string w = $"vision.layers.{il}.";
            var h = Norm(b, p, x, w + "input_norm");
            Value qk;
            Value v;
            if (b.HasWeight(w + "attention.qkv.weight"))
            {
                var qkv = b.Linear(h, w + "attention.qkv.weight", w + "attention.qkv.bias");          // [N, 3H]: q | k | v
                qk = b.Reshape(b.Slice(qkv, 1, 0, 2L * H), 0, 2 * heads, D);                       // [N, 2·heads, D]
                v = b.Reshape(b.Slice(qkv, 1, 2L * H, H), 0, heads, D);
            }
            else
            {
                var q0 = b.Reshape(b.Linear(h, w + "attention.q.weight", w + "attention.q.bias"), 0, heads, D);
                var k0 = b.Reshape(b.Linear(h, w + "attention.k.weight", w + "attention.k.bias"), 0, heads, D);
                qk = b.Concat(1, q0, k0);
                v = b.Reshape(b.Linear(h, w + "attention.v.weight", w + "attention.v.bias"), 0, heads, D);
            }

            if (cos is not null)
            {
                qk = Rotate(b, qk, cos, sinSigned!, D);
            }

            var q = b.Slice(qk, 1, 0, heads);
            var k = b.Slice(qk, 1, heads, heads);
            bool windowed = windowSpans is not null && Array.IndexOf(p.FullAttentionLayers, il) < 0;
            var attn = b.Emit1("Attention", windowed ? [q, k, v, windowSpans!] : [q, k, v], Attributes.From(("scale", 1f / MathF.Sqrt(D)), ("causal", false)), $"{w}attn");
            var o = b.Linear(b.Reshape(attn, 0, H), w + "attention.o.weight", w + "attention.o.bias");
            x = b.Add(x, o, $"{w}attn_residual");

            h = Norm(b, p, x, w + "post_attention_norm");
            var up = b.Linear(h, w + "mlp.up.weight", w + "mlp.up.bias");
            var act = b.HasWeight(w + "mlp.gate.weight")
                ? b.Mul(Activation(b, b.Linear(h, w + "mlp.gate.weight", w + "mlp.gate.bias"), p.Activation), up)
                : Activation(b, up, p.Activation);
            var down = b.Linear(act, w + "mlp.down.weight", w + "mlp.down.bias");
            x = b.Add(x, down, $"{w}l_out");
            if (Array.IndexOf(p.DeepstackLayers, il) >= 0)
            {
                string d = $"vision.deepstack.{il}.";
                var merged = Norm(b, p, Merge(b, x, p), d + "norm");
                var dsUp = b.Linear(merged, d + "up.weight", d + "up.bias");
                deepstack.Add(b.Linear(Activation(b, dsUp, p.MergerActivation), d + "down.weight", d + "down.bias", $"{d}out"));
            }
        }

        using (b.Group("merger"))
        {
            var h = Merge(b, Norm(b, p, x, "vision.merger.norm"), p);
            var up = b.Linear(h, "vision.merger.up.weight", "vision.merger.up.bias");
            var down = b.Linear(Activation(b, up, p.MergerActivation), "vision.merger.down.weight", "vision.merger.down.bias", VisionParameters.EmbeddingsOutput);
            var rows = deepstack.Count == 0 ? down : b.Concat(1, [down, .. deepstack]);
            // Windowed towers run window by window; the output goes back to merge-window raster order.
            b.Output(VisionParameters.EmbeddingsOutput, windowOrder is null ? rows : b.Gather(rows, windowOrder));
        }
        return b.Build();
    }

    // m×m neighbouring patches (consecutive in merge-window order) → one row of m²·hidden.
    private static Value Merge(GraphBuilder b, Value x, VisionParameters p) => p.MergeUnit == 1 ? x
        : b.Emit1("Reshape", [x], Attributes.From(("shape", new long[] { 0, (long)p.MergeUnit * p.HiddenSize }), ("outer_symbol", VisionParameters.MergedSymbol)));

    private static Value Norm(GraphBuilder b, VisionParameters p, Value x, string name) => p.NormType == "rms"
        ? b.RmsNorm(x, b.Weight(name + ".weight"), p.NormEps)
        : b.Emit1("LayerNorm", [x, b.Weight(name + ".weight"), b.Weight(name + ".bias")], Attributes.From(("eps", p.NormEps)));

    private static Value Activation(GraphBuilder b, Value x, string kind) => kind switch
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
        var freqs = new float[F];
        for (int j = 0; j < F; j++)
        {
            freqs[j] = (float)Math.Pow(p.RopeBase, -2.0 * j / (D / 2));
        }

        var signs = new float[D];
        for (int i = 0; i < D; i++)
        {
            signs[i] = i < D / 2 ? -1f : 1f;
        }

        var cf = b.Constant("rope_freqs", DataType.F32, new Shape(1, F), Bytes(freqs));
        var cs = b.Constant("rope_signs", DataType.F32, new Shape(1, 1, D), Bytes(signs));

        var pos = b.Emit1("Cast", [positions], Attributes.From(("type", DataType.F32.Name)));
        var row = b.Mul(b.Slice(pos, 1, 0, 1), cf);
        var col = b.Mul(b.Slice(pos, 1, 1, 1), cf);
        var half = b.Concat(1, row, col);
        var angle = b.Reshape(b.Concat(1, half, half), 0, 1, D);
        var cos = b.Unary("Cos", angle, "rope_cos");
        var sin = b.Mul(b.Unary("Sin", angle), cs, "rope_sin");
        return (cos, sin);
    }

    private static Value Rotate(GraphBuilder b, Value x, Value cos, Value sinSigned, int D)
    {
        var swapped = b.Concat(2, b.Slice(x, 2, D / 2, D / 2), b.Slice(x, 2, 0, D / 2));
        return b.Add(b.Mul(x, cos), b.Mul(swapped, sinSigned));
    }

    private static byte[] Bytes(float[] v)
    {
        var r = new byte[v.Length * 4];
        Buffer.BlockCopy(v, 0, r, 0, r.Length);
        return r;
    }
}
