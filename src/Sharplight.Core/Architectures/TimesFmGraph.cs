using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;
using Sharplight.TimeSeries;

namespace Sharplight.Architectures;

/// <summary>
/// Emits a TimesFM forecaster as <see cref="TimeSeriesGraphs.PatchDecoderGraph"/> (the patch-decoder form of <see cref="TimeSeriesGraphs"/>): the input
/// residual block embeds every patch (its normalized values and padding indicators), a causal transformer runs over the patches, and the output residual
/// blocks turn the hidden state of the last patch into the normalized forecast of the next output patch.
/// <list type="bullet">
/// <item><description>TimesFM 1 / 2: the frequency embedding is added to every patch; layers are RMS-normed attention (per-dimension query scale, no
/// positions) and a LayerNorm-ed ReLU feed-forward block, each with a plain residual.</description></item>
/// <item><description>TimesFM 2.5: rotary positions, query and key RMS norms after the rotation, the per-dimension query scale, RMS norms before and after
/// both the attention and the SiLU feed-forward block; a second, continuous quantile head.</description></item>
/// </list>
/// </summary>
internal static class TimesFmGraph
{
    /// <summary>Builds the graph.</summary>
    /// <param name="p">The hyper-parameters.</param>
    /// <param name="weights">Canonical weight lookup.</param>
    public static ComputeGraph Build(TimesFmParameters p, Func<string, WeightInfo?> weights)
    {
        var b = new GraphBuilder(TimeSeriesGraphs.PatchDecoderGraph, weights);
        var rows = Dim.Sym(TimeSeriesGraphs.PatchSymbol);
        int ps = p.PatchSize;
        var patches = b.Input(TimeSeriesGraphs.PatchesInput, InputRole.Other, DataType.F32, new Shape(rows, 2 * ps));
        var positions = p.Is25 ? b.Input(TimeSeriesGraphs.PositionsInput, InputRole.Positions, DataType.I32, new Shape(rows, 1)) : null;
        var spans = b.Input(TimeSeriesGraphs.SpansInput, InputRole.RowSpans, DataType.I32, new Shape(rows, 2));
        var lastRow = b.Input(TimeSeriesGraphs.LastRowInput, InputRole.Other, DataType.I32, new Shape(1));
        Value x;
        using (b.Group("input"))
        {
            // The models read padding indicators (1 = missing), the contract carries the observation mask.
            var tokens = b.Concat(1, b.Slice(patches, 1, 0, ps), b.Scale(b.Slice(patches, 1, ps, ps), -1f, 1f));
            x = ResidualBlock(b, tokens, "input_ff_layer");
            if (!p.Is25)
            {
                x = b.Add(x, b.Gather(b.Weight("freq_emb.weight"), Index(b, "frequency_id", p.Frequency)));
            }
        }

        for (int layer = 0; layer < p.LayerCount; layer++)
        {
            using var _ = b.Group($"layers.{layer}");
            string n = $"layers.{layer}.";
            x = p.Is25 ? Layer25(b, x, n, positions!, spans, p) : Layer(b, x, n, spans, p);
        }

        using (b.Group("output"))
        {
            var last = b.Gather(x, lastRow);
            if (p.Is25)
            {
                b.Output(TimeSeriesGraphs.PointOutput, b.Reshape(ResidualBlock(b, last, "output_projection_point"), p.PredictionLength, p.Columns));
                if (p.QuantileHorizon > 0)
                {
                    b.Output(TimeSeriesGraphs.QuantileSpreadOutput, b.Reshape(ResidualBlock(b, last, "output_projection_quantiles"), p.QuantileHorizon, p.Columns));
                }
            }
            else
            {
                b.Output(TimeSeriesGraphs.PointOutput, b.Reshape(ResidualBlock(b, last, "horizon_ff_layer"), p.PredictionLength, p.Columns));
            }
        }

        return b.Build();
    }

    // TimesFM 1 / 2: x + attention(rms(x)), then x + down(relu(gate(layernorm(x)))).
    private static Value Layer(GraphBuilder b, Value x, string n, Value spans, TimesFmParameters p)
    {
        var h = b.RmsNorm(x, b.Weight(n + "input_layernorm.weight"), p.NormEps);
        string a = n + "self_attn.";
        var q = ScaleQuery(b, Heads(b, Linear(b, h, a + "q_proj"), p), a, p);
        var k = Heads(b, Linear(b, h, a + "k_proj"), p);
        var v = Heads(b, Linear(b, h, a + "v_proj"), p);
        x = b.Add(x, Linear(b, Attend(b, q, k, v, spans, p), a + "o_proj"));
        var g = b.Emit1("LayerNorm", [x, b.Weight(n + "mlp.layer_norm.weight"), b.Weight(n + "mlp.layer_norm.bias")], Attributes.From(("eps", 1e-6f)));
        return b.Add(x, Linear(b, b.Unary("Relu", Linear(b, g, n + "mlp.gate_proj")), n + "mlp.down_proj"));
    }

    // TimesFM 2.5: x + rms(attention(rms(x))), then x + rms(ff1(silu(ff0(rms(x))))).
    private static Value Layer25(GraphBuilder b, Value x, string n, Value positions, Value spans, TimesFmParameters p)
    {
        var h = b.RmsNorm(x, b.Weight(n + "input_layernorm.weight"), p.NormEps);
        string a = n + "self_attn.";
        Value q;
        Value k;
        Value v;
        int inner = p.HeadCount * p.HeadDim;
        if (b.TryWeight(a + "qkv_proj.weight") is not null)
        {
            var qkv = Linear(b, h, a + "qkv_proj");
            q = b.Slice(qkv, 1, 0, inner);
            k = b.Slice(qkv, 1, inner, inner);
            v = b.Slice(qkv, 1, 2 * inner, inner);
        }
        else
        {
            q = Linear(b, h, a + "q_proj");
            k = Linear(b, h, a + "k_proj");
            v = Linear(b, h, a + "v_proj");
        }

        var rope = Attributes.From(("dims", (long)p.HeadDim), ("base", p.RopeBase), ("style", "neox"));
        q = b.RmsNorm(b.Emit1("Rope", [Heads(b, q, p), positions], rope), b.Weight(a + "q_norm.weight"), p.NormEps);
        k = b.RmsNorm(b.Emit1("Rope", [Heads(b, k, p), positions], rope), b.Weight(a + "k_norm.weight"), p.NormEps);
        q = ScaleQuery(b, q, a, p);
        var attended = Linear(b, Attend(b, q, k, Heads(b, v, p), spans, p), a + "o_proj");
        x = b.Add(x, b.RmsNorm(attended, b.Weight(n + "post_attention_layernorm.weight"), p.NormEps));
        var f = b.RmsNorm(x, b.Weight(n + "pre_feedforward_layernorm.weight"), p.NormEps);
        f = Linear(b, b.Silu(Linear(b, f, n + "mlp.ff0")), n + "mlp.ff1");
        return b.Add(x, b.RmsNorm(f, b.Weight(n + "post_feedforward_layernorm.weight"), p.NormEps));
    }

    // The learned per-dimension query scale: softplus(scaling) · 1.442695041 / √D (the scores are not scaled again).
    private static Value ScaleQuery(GraphBuilder b, Value q, string a, TimesFmParameters p)
        => b.Mul(q, b.Scale(b.Softplus(b.Weight(a + "scaling")), 1.442695041f / MathF.Sqrt(p.HeadDim)));

    private static Value Attend(GraphBuilder b, Value q, Value k, Value v, Value spans, TimesFmParameters p)
        => b.Reshape(b.Emit1("Attention", [q, k, v, spans], Attributes.From(("scale", 1f), ("causal", true))), 0, p.HeadCount * p.HeadDim);

    private static Value Heads(GraphBuilder b, Value x, TimesFmParameters p) => b.Reshape(x, 0, p.HeadCount, p.HeadDim);

    // The residual block of every TimesFM: output(silu(input(x))) + residual(x), with biases where the checkpoint has them.
    private static Value ResidualBlock(GraphBuilder b, Value x, string module)
        => b.Add(Linear(b, b.Silu(Linear(b, x, module + ".input_layer")), module + ".output_layer"), Linear(b, x, module + ".residual_layer"));

    private static Value Linear(GraphBuilder b, Value x, string module)
    {
        var y = b.Emit1("MatMul", [x, b.Weight(module + ".weight")], Attributes.From((StandardOps.F32ActivationsAttribute, true)));
        return b.TryWeight(module + ".bias") is { } bias ? b.Add(y, bias) : y;
    }

    private static Value Index(GraphBuilder b, string name, int index)
        => b.Constant(name, DataType.I32, new Shape(1), MemoryMarshal.AsBytes<int>([index]).ToArray());
}
