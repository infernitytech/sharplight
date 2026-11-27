using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;
using Sharplight.TimeSeries;

namespace Sharplight.Architectures;

/// <summary>
/// Emits the Chronos-Bolt forecaster as one graph of generic ops (the patch-quantile form of <see cref="TimeSeriesGraphs"/>) over a batch of series (the symbolic
/// <see cref="TimeSeriesGraphs.SeriesSymbol"/> axis leads every tensor): the input residual block embeds every patch, the register token row follows, the T5 encoder
/// runs over all rows of each series with the patches without observations masked out as keys, and a single decoder step from the start token (whose
/// self-attention over its one row reduces to the value projection) cross-attends to the encoder output of its series; the output residual block turns its hidden
/// state into the scaled quantile forecasts.
/// </summary>
internal static class ChronosBoltGraph
{
    /// <summary>Builds the graph.</summary>
    /// <param name="p">The hyper-parameters.</param>
    /// <param name="weights">Canonical weight lookup.</param>
    public static ComputeGraph Build(ChronosBoltParameters p, Func<string, WeightInfo?> weights)
    {
        var b = new GraphBuilder(TimeSeriesGraphs.ForecastGraph, weights);
        var hp = p.Encoder;
        int patches = p.Patches;
        int rows = hp.ContextLength;
        var series = Dim.Sym(TimeSeriesGraphs.SeriesSymbol);
        var input = b.Input(TimeSeriesGraphs.PatchesInput, InputRole.Other, DataType.F32, new Shape(series, patches, 2 * p.PatchSize));
        var mask = b.Input(TimeSeriesGraphs.MaskInput, InputRole.Other, DataType.F32, new Shape(series, patches));

        // [S, 1, 1] ones (from the mask's first column: finite, so 0 · m + 1 is exactly 1), to broadcast the shared token rows over the series.
        var ones = b.Reshape(b.Scale(b.Slice(mask, 1, 0, 1), 0f, 1f), 0, 1, 1);
        Value x;
        Value keyMask;
        using (b.Group("input"))
        {
            x = ResidualBlock(b, input, "input_patch_embedding", p.PatchActivation);
            if (p.RegisterToken)
            {
                x = b.Concat(1, x, b.Mul(ones, Row(b, "register_token", 1)));
                mask = b.Concat(1, mask, b.Reshape(ones, 0, 1));
            }

            // (mask − 1) · 10⁹: 0 for observed patches, a large negative score for empty ones, broadcast over heads and query rows.
            keyMask = b.Reshape(b.Scale(mask, 1e9f, -1e9f), 0, 1, 1, rows);
        }

        var encoded = T5TextGraphBuilder.Encode(b, x, keyMask, hp, "");
        var y = b.Mul(ones, Row(b, "decoder_start", 0));
        int inner = hp.HeadCount * hp.HeadDim;
        for (int layer = 0; layer < p.DecoderLayers; layer++)
        {
            using var _ = b.Group($"decoder.layers.{layer}");
            string d = $"decoder.layers.{layer}.";
            var h = b.RmsNorm(y, b.Weight(d + "attention_norm.weight"), hp.NormEps);
            y = b.Add(y, Linear(b, Linear(b, h, d + "attention.v.weight"), d + "attention.o.weight"));
            h = b.RmsNorm(y, b.Weight(d + "cross_attention_norm.weight"), hp.NormEps);
            var q = b.Reshape(Linear(b, h, d + "cross_attention.q.weight"), 0, hp.HeadCount, 1, hp.HeadDim);
            var k = Heads(b, Linear(b, encoded, d + "cross_attention.k.weight"), rows, hp);
            var v = Heads(b, Linear(b, encoded, d + "cross_attention.v.weight"), rows, hp);
            var scores = b.Add(b.Emit1("BatchMatMul", [q, k], Attributes.From(("trans_b", true))), keyMask);
            var attended = b.Emit1("BatchMatMul", [b.Emit1("Softmax", [scores]), v]);
            y = b.Add(y, Linear(b, b.Reshape(attended, 0, 1, inner), d + "cross_attention.o.weight"));
            h = b.RmsNorm(y, b.Weight(d + "mlp_norm.weight"), hp.NormEps);
            y = b.Add(y, Linear(b, Activate(b, Linear(b, h, d + "mlp.up.weight"), hp.Activation), d + "mlp.down.weight"));
        }

        using (b.Group("output"))
        {
            y = b.RmsNorm(y, b.Weight("decoder.output_norm.weight"), hp.NormEps);
            var forecast = ResidualBlock(b, y, "output_patch_embedding", p.PatchActivation);
            b.Output(TimeSeriesGraphs.QuantilesOutput, b.Reshape(forecast, 0, p.Quantiles.Length, p.PredictionLength));
        }

        return b.Build();
    }

    // Row `index` of the two-row token embedding (0: decoder start, 1: register token) as [1, 1, hidden].
    private static Value Row(GraphBuilder b, string name, int index)
    {
        var row = b.Gather(b.Weight("token_embedding.weight"), b.Constant(name + "_id", DataType.I32, new Shape(1), MemoryMarshal.AsBytes<int>([index]).ToArray()));
        return b.Reshape(row, 1, 1, -1);
    }

    // [S, rows, hidden] → [S, heads, rows, head dim].
    private static Value Heads(GraphBuilder b, Value x, int rows, T5TextHyperParameters hp)
        => b.Contiguous(b.Permute(b.Reshape(x, 0, rows, hp.HeadCount, hp.HeadDim), 0, 2, 1, 3));

    // Chronos-Bolt's ResidualBlock: output(act(hidden(x))) + residual(x).
    private static Value ResidualBlock(GraphBuilder b, Value x, string module, string activation)
    {
        var hidden = Activate(b, b.Linear(x, module + ".hidden_layer.weight", module + ".hidden_layer.bias"), activation);
        return b.Add(b.Linear(hidden, module + ".output_layer.weight", module + ".output_layer.bias"),
            b.Linear(x, module + ".residual_layer.weight", module + ".residual_layer.bias"));
    }

    private static Value Linear(GraphBuilder b, Value x, string weight)
        => b.Emit1("MatMul", [x, b.Weight(weight)], Attributes.From((StandardOps.F32ActivationsAttribute, true)));

    private static Value Activate(GraphBuilder b, Value x, string activation) => activation switch
    {
        "relu" => b.Unary("Relu", x),
        "gelu" => b.Unary("Gelu", x),
        "gelu_tanh" => b.Unary("GeluTanh", x),
        "silu" => b.Silu(x),
        _ => throw new NotSupportedException($"chronos_bolt: unknown activation '{activation}'."),
    };
}
