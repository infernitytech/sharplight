using Sharplight.Graph;

namespace Sharplight.Architectures;

/// <summary>Graph pieces of pre-norm transformer towers: layer norm, activations and the encoder layer stack with optional LayerScale.</summary>
internal static class TowerBlocks
{
    /// <summary>A LayerNorm with the weights <c>name.weight</c> and <c>name.bias</c>.</summary>
    public static Value LayerNorm(GraphBuilder b, Value v, string name, float eps)
        => b.Emit1("LayerNorm", [v, b.Weight(name + ".weight"), b.Weight(name + ".bias")], Attributes.From(("eps", eps)));

    /// <summary>The named MLP activation: gelu (erf), gelu_tanh, quick_gelu, relu or silu.</summary>
    public static Value Activation(GraphBuilder b, Value t, string act) => act switch
    {
        "gelu" => b.Unary("Gelu", t),
        "gelu_tanh" => b.Unary("GeluTanh", t),
        "quick_gelu" => b.Mul(t, b.Sigmoid(b.Scale(t, 1.702f))),
        "relu" => b.Unary("Relu", t),
        "silu" => b.Silu(t),
        _ => throw new NotSupportedException($"Unknown mlp.activation '{act}'."),
    };

    // Causal attention of a fixed-length sequence as an additive f32 [rows, rows] mask (the attention op's causal form needs packed-sequence spans).
    private static Value CausalMask(GraphBuilder b, int rows)
    {
        var mask = new float[rows * rows];
        for (int i = 0; i < rows; i++)
        {
            for (int j = i + 1; j < rows; j++)
            {
                mask[i * rows + j] = -1e9f;
            }
        }

        return b.Constant("causal_mask", Tensors.DataType.F32, new Tensors.Shape(rows, rows), System.Runtime.InteropServices.MemoryMarshal.AsBytes(mask.AsSpan()).ToArray());
    }

    /// <summary>Multiplies by the LayerScale weight when the model has it.</summary>
    public static Value Scaled(GraphBuilder b, Value v, string scale) => b.HasWeight(scale) ? b.Mul(v, b.Weight(scale)) : v;

    /// <summary>
    /// The layer stack over <paramref name="x"/> [rows, hidden]: <c>{Prefix}layers.{i}.</c> with <c>input_norm</c>, <c>attention.{q,k,v,o}</c>,
    /// optional <c>layer_scale_1/2</c>, <c>post_attention_norm</c> and <c>mlp.{up,down}</c>.
    /// </summary>
    /// <param name="b">The graph builder.</param>
    /// <param name="x">The input rows [rows, hidden].</param>
    /// <param name="t">The tower hyper-parameters.</param>
    /// <param name="rows">The number of rows.</param>
    /// <param name="causal">Whether attention is causal.</param>
    /// <param name="taps">Receives the output of every layer listed in <paramref name="tapAfter"/> (1-based layer counts), in order; or <see langword="null"/>.</param>
    /// <param name="tapAfter">The layer counts whose outputs are tapped (intermediate features for dense-prediction necks), or <see langword="null"/>.</param>
    /// <param name="layerOffset">The index of the first layer, to build a few layers of the stack at a time.</param>
    public static Value Layers(GraphBuilder b, Value x, TowerOptions t, int rows, bool causal, List<Value>? taps = null, IReadOnlyCollection<int>? tapAfter = null, int layerOffset = 0)
    {
        Value? mask = causal ? CausalMask(b, rows) : null;
        for (int il = layerOffset; il < layerOffset + t.Layers; il++)
        {
            using var _ = b.Group($"{t.Prefix}layers.{il}");
            string p = $"{t.Prefix}layers.{il}.";
            var h = LayerNorm(b, x, p + "input_norm", t.Eps);
            var q = b.Reshape(b.Linear(h, p + "attention.q.weight", p + "attention.q.bias"), rows, t.Heads, t.HeadSize);
            var k = b.Reshape(b.Linear(h, p + "attention.k.weight", p + "attention.k.bias"), rows, t.Heads, t.HeadSize);
            var v = b.Reshape(b.Linear(h, p + "attention.v.weight", p + "attention.v.bias"), rows, t.Heads, t.HeadSize);
            var attn = b.Emit1("Attention", mask is null ? [q, k, v] : [q, k, v, mask], Attributes.From(("scale", 1f / MathF.Sqrt(t.HeadSize)), ("causal", false)), $"{p}attn");
            var o = b.Linear(b.Reshape(attn, rows, t.Hidden), p + "attention.o.weight", p + "attention.o.bias");
            x = b.Add(x, Scaled(b, o, p + "layer_scale_1.weight"), $"{p}attn_residual");
            h = LayerNorm(b, x, p + "post_attention_norm", t.Eps);
            var mlp = b.Linear(Activation(b, b.Linear(h, p + "mlp.up.weight", p + "mlp.up.bias"), t.Activation), p + "mlp.down.weight", p + "mlp.down.bias");
            x = b.Add(x, Scaled(b, mlp, p + "layer_scale_2.weight"), $"{p}l_out");
            if (taps is not null && tapAfter is not null && tapAfter.Contains(il + 1))
            {
                taps.Add(x);
            }
        }

        return x;
    }
}
