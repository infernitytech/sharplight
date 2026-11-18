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
    /// <param name="rotate">Position rotation applied to the queries and keys [rows, heads, head size] (see <see cref="PatchRope"/>), or <see langword="null"/>.</param>
    public static Value Layers(GraphBuilder b, Value x, TowerOptions t, int rows, bool causal, List<Value>? taps = null, IReadOnlyCollection<int>? tapAfter = null, int layerOffset = 0, Func<Value, Value>? rotate = null)
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
            if (rotate is not null)
            {
                q = rotate(q);
                k = rotate(k);
            }

            var attn = b.Emit1("Attention", mask is null ? [q, k, v] : [q, k, v, mask], Attributes.From(("scale", 1f / MathF.Sqrt(t.HeadSize)), ("causal", false)), $"{p}attn");
            var o = b.Linear(b.Reshape(attn, rows, t.Hidden), p + "attention.o.weight", p + "attention.o.bias");
            x = b.Add(x, Scaled(b, o, p + "layer_scale_1.weight"), $"{p}attn_residual");
            h = LayerNorm(b, x, p + "post_attention_norm", t.Eps);
            var up = b.Linear(h, p + "mlp.up.weight", p + "mlp.up.bias");
            var hiddenAct = b.HasWeight(p + "mlp.gate.weight")
                ? b.Mul(Activation(b, b.Linear(h, p + "mlp.gate.weight", p + "mlp.gate.bias"), t.Activation), up)
                : Activation(b, up, t.Activation);
            var mlp = b.Linear(hiddenAct, p + "mlp.down.weight", p + "mlp.down.bias");
            x = b.Add(x, Scaled(b, mlp, p + "layer_scale_2.weight"), $"{p}l_out");
            if (taps is not null && tapAfter is not null && tapAfter.Contains(il + 1))
            {
                taps.Add(x);
            }
        }

        return x;
    }

    /// <summary>
    /// DINOv3's two-dimensional rotary position embedding over the patch rows of a square grid: patch centers in [-1, 1] on both axes, the angles
    /// <c>2π · coordinate · θ^(-4k / head size)</c> for both axes laid out as <c>[y, x, y, x]</c> over the head size, applied as <c>x · cos + rotate_half(x) · sin</c>
    /// to the patch rows only (the class and register rows pass through). The tables are constants of the graph.
    /// </summary>
    /// <param name="b">The graph builder.</param>
    /// <param name="grid">Patches per side.</param>
    /// <param name="headSize">The attention head size.</param>
    /// <param name="theta">The rotary base.</param>
    /// <param name="rows">The total number of rows (prefix rows + <c>grid²</c>).</param>
    public static Func<Value, Value> PatchRope(GraphBuilder b, int grid, int headSize, float theta, int rows)
    {
        int n = grid * grid;
        int quarter = headSize / 4;
        var cos = new float[n * headSize];
        var sin = new float[n * headSize];
        for (int py = 0; py < grid; py++)
        {
            for (int px = 0; px < grid; px++)
            {
                int row = (py * grid) + px;
                float[] coords = [(2f * ((py + 0.5f) / grid)) - 1f, (2f * ((px + 0.5f) / grid)) - 1f];
                for (int axis = 0; axis < 2; axis++)
                {
                    for (int k = 0; k < quarter; k++)
                    {
                        double invFreq = 1.0 / Math.Pow(theta, 4.0 * k / headSize);
                        double angle = 2.0 * Math.PI * coords[axis] * invFreq;
                        int d = (axis * quarter) + k;
                        foreach (int offset in new[] { 0, headSize / 2 })
                        {
                            cos[(row * headSize) + d + offset] = (float)Math.Cos(angle);
                            sin[(row * headSize) + d + offset] = (float)Math.Sin(angle);
                        }
                    }
                }
            }
        }

        var shape = new Tensors.Shape(n, 1, headSize);
        var cosValue = b.Constant("rope_cos", Tensors.DataType.F32, shape, System.Runtime.InteropServices.MemoryMarshal.AsBytes(cos.AsSpan()).ToArray());
        var sinValue = b.Constant("rope_sin", Tensors.DataType.F32, shape, System.Runtime.InteropServices.MemoryMarshal.AsBytes(sin.AsSpan()).ToArray());
        int half = headSize / 2;
        return t =>
        {
            var patches = b.Slice(t, 0, rows - n, n);
            var rotated = b.Concat(2, b.Scale(b.Slice(patches, 2, half, half), -1f), b.Slice(patches, 2, 0, half));
            var turned = b.Add(b.Mul(patches, cosValue), b.Mul(rotated, sinValue));
            return rows == n ? turned : b.Concat(0, b.Slice(t, 0, 0, rows - n), turned);
        };
    }
}
