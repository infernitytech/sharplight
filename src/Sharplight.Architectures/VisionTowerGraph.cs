using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;
using Sharplight.Vision;

namespace Sharplight.Architectures;

/// <summary>
/// Emits the <see cref="WellKnown.VisionEncodeGraph"/> of a ViT-style vision tower described by the canonical
/// <see cref="VisionParameters"/> and <c>vision.*</c> weights (documented on <see cref="VisionParameters"/>):
/// patch embedding (a MatMul over the patchified input) + learned position embedding (bilinear taps) → pre-norm
/// transformer blocks (LayerNorm, fused-QKV bidirectional multi-head attention with axial 2-D rotary embedding, MLP)
/// → patch merger (LayerNorm, m×m merge, 2-layer MLP) producing embeddings of the language model's hidden size.
/// Shared by every architecture that uses such a tower; it does not know which file the weights came from.
/// </summary>
public static class VisionTowerGraph
{
    public static ComputeGraph Build(VisionParameters p, Func<string, WeightInfo?> weights)
    {
        var b = new GraphBuilder(WellKnown.VisionEncodeGraph, weights);
        var N = Dim.Sym(VisionParameters.PatchSymbol);
        int H = p.HiddenSize, heads = p.HeadCount, D = p.HeadDim;
        float eps = p.NormEps;

        Value x;
        Value positions;
        Value embIndex, embWeight;
        using (b.Group("input"))
        {
            var patches = b.Input(VisionParameters.PatchesInput, InputRole.Media, DataType.F32, new Shape(N, p.PatchDim));
            positions = b.Input(VisionParameters.PositionsInput, InputRole.Positions, DataType.I32, new Shape(N, 2));
            embIndex = b.Input(VisionParameters.EmbeddingIndexInput, InputRole.Other, DataType.I32, new Shape(N, 4));
            embWeight = b.Input(VisionParameters.EmbeddingWeightInput, InputRole.Other, DataType.F32, new Shape(N, 4));
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

        Value? cos = null, sinSigned = null;
        if (p.RopeBase > 0) (cos, sinSigned) = RotaryTables(b, p, positions);

        for (int il = 0; il < p.LayerCount; il++)
        {
            using var _ = b.Group($"vision.layers.{il}");
            string w = $"vision.layers.{il}.";
            var h = LayerNorm(b, x, w + "input_norm", eps);
            var qkv = b.Linear(h, w + "attention.qkv.weight", w + "attention.qkv.bias");          // [N, 3H]: q | k | v
            var qk = b.Reshape(b.Slice(qkv, 1, 0, 2L * H), 0, 2 * heads, D);                       // [N, 2·heads, D]
            if (cos is not null) qk = Rotate(b, qk, cos, sinSigned!, D);
            var q = b.Slice(qk, 1, 0, heads);
            var k = b.Slice(qk, 1, heads, heads);
            var v = b.Reshape(b.Slice(qkv, 1, 2L * H, H), 0, heads, D);
            var attn = b.Emit1("Attention", [q, k, v], Attributes.From(("scale", 1f / MathF.Sqrt(D)), ("causal", false)), $"{w}attn");
            var o = b.Linear(b.Reshape(attn, 0, H), w + "attention.o.weight", w + "attention.o.bias");
            x = b.Add(x, o, $"{w}attn_residual");

            h = LayerNorm(b, x, w + "post_attention_norm", eps);
            var up = b.Linear(h, w + "mlp.up.weight", w + "mlp.up.bias");
            var down = b.Linear(Activation(b, up, p.Activation), w + "mlp.down.weight", w + "mlp.down.bias");
            x = b.Add(x, down, $"{w}l_out");
        }

        using (b.Group("merger"))
        {
            var h = LayerNorm(b, x, "vision.merger.norm", eps);
            if (p.MergeUnit > 1)
                h = b.Emit1("Reshape", [h], Attributes.From(("shape", new long[] { 0, (long)p.MergeUnit * H }), ("outer_symbol", VisionParameters.MergedSymbol)));
            var up = b.Linear(h, "vision.merger.up.weight", "vision.merger.up.bias");
            var down = b.Linear(Activation(b, up, p.MergerActivation), "vision.merger.down.weight", "vision.merger.down.bias", VisionParameters.EmbeddingsOutput);
            b.Output(VisionParameters.EmbeddingsOutput, down);
        }
        return b.Build();
    }

    static Value LayerNorm(GraphBuilder b, Value x, string name, float eps)
        => b.Emit1("LayerNorm", [x, b.Weight(name + ".weight"), b.Weight(name + ".bias")], Attributes.From(("eps", eps)));

    static Value Activation(GraphBuilder b, Value x, string kind) => kind switch
    {
        "gelu_tanh" => b.Unary("GeluTanh", x),
        "gelu" => b.Unary("Gelu", x),
        "silu" => b.Silu(x),
        _ => throw new NotSupportedException($"Vision activation '{kind}'."),
    };

    /// <summary>
    /// Axial 2-D rotary tables for positions (row, col): per axis head_dim/4 frequencies base^(-2j/(head_dim/2)); the angles
    /// [row·f | col·f] are repeated to span the head (rotation pairs (i, i + head_dim/2)). Returns cos and the sign-folded
    /// sin (negated on the first half) as [N, 1, D], so that rotate(x) = x·cos + swap_halves(x)·sinSigned.
    /// </summary>
    static (Value Cos, Value SinSigned) RotaryTables(GraphBuilder b, VisionParameters p, Value positions)
    {
        int D = p.HeadDim, F = D / 4;
        var freqs = new float[F];
        for (int j = 0; j < F; j++) freqs[j] = (float)Math.Pow(p.RopeBase, -2.0 * j / (D / 2));
        var signs = new float[D];
        for (int i = 0; i < D; i++) signs[i] = i < D / 2 ? -1f : 1f;
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

    static Value Rotate(GraphBuilder b, Value x, Value cos, Value sinSigned, int D)
    {
        var swapped = b.Concat(2, b.Slice(x, 2, D / 2, D / 2), b.Slice(x, 2, 0, D / 2));
        return b.Add(b.Mul(x, cos), b.Mul(swapped, sinSigned));
    }

    static byte[] Bytes(float[] v)
    {
        var r = new byte[v.Length * 4];
        Buffer.BlockCopy(v, 0, r, 0, r.Length);
        return r;
    }
}
