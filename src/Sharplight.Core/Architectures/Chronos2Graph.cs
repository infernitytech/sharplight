using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;
using Sharplight.TimeSeries;

namespace Sharplight.Architectures;

/// <summary>
/// Emits the Chronos-2 forecaster (the group-encoder form of <see cref="TimeSeriesGraphs"/>): the input residual block embeds every patch row (the register
/// rows take the register token instead), then every layer runs pre-norm attention along time within each series (rotary positions, unscaled scores), pre-norm
/// attention across the series of a group at each position (no positions), and a pre-norm ReLU feed-forward block; after the final RMS norm the output
/// residual block turns the read rows into the scaled quantile forecasts.
/// </summary>
/// <remarks>
/// Both attentions are packed-span attentions over all rows; the key bias of empty context patches rides in an extra query / key column (query 1, key the
/// bias), which adds it to every score of that key exactly. The attention across series runs in the group order (<see cref="TimeSeriesGraphs.GroupOrderInput"/>).
/// </remarks>
internal static class Chronos2Graph
{
    // The extra query / key columns that carry the key bias (one used, padded to a multiple of 8 for the attention kernels).
    private const int BiasColumns = 8;

    /// <summary>Builds the graph.</summary>
    /// <param name="p">The hyper-parameters.</param>
    /// <param name="weights">Canonical weight lookup.</param>
    public static ComputeGraph Build(Chronos2Parameters p, Func<string, WeightInfo?> weights)
    {
        var b = new GraphBuilder(TimeSeriesGraphs.GroupEncoderGraph, weights);
        var rows = Dim.Sym(TimeSeriesGraphs.PatchSymbol);
        var patches = b.Input(TimeSeriesGraphs.PatchesInput, InputRole.Other, DataType.F32, new Shape(rows, 3 * p.PatchSize));
        var register = b.Input(TimeSeriesGraphs.RegisterRowsInput, InputRole.Other, DataType.F32, new Shape(rows, 1));
        var keyBias = b.Input(TimeSeriesGraphs.KeyBiasInput, InputRole.Other, DataType.F32, new Shape(rows, 1));
        var positions = b.Input(TimeSeriesGraphs.PositionsInput, InputRole.Positions, DataType.I32, new Shape(rows, 1));
        var spans = b.Input(TimeSeriesGraphs.SpansInput, InputRole.RowSpans, DataType.I32, new Shape(rows, 2));
        var order = b.Input(TimeSeriesGraphs.GroupOrderInput, InputRole.Other, DataType.I32, new Shape(rows));
        var inverse = b.Input(TimeSeriesGraphs.GroupInverseInput, InputRole.Other, DataType.I32, new Shape(rows));
        var groupSpans = b.Input(TimeSeriesGraphs.GroupSpansInput, InputRole.Other, DataType.I32, new Shape(rows, 2));
        var read = b.Input(TimeSeriesGraphs.ReadRowsInput, InputRole.Other, DataType.I32, new Shape(Dim.Sym(TimeSeriesGraphs.ReadSymbol)));
        Value x;
        Value queryColumns;
        Value keyColumns;
        Value groupKeyColumns;
        using (b.Group("input"))
        {
            x = ResidualBlock(b, patches, "input_patch_embedding", p.Activation);
            if (p.RegisterToken)
            {
                var token = b.Gather(b.Weight("token_embedding.weight"), b.Constant("register_token_id", DataType.I32, new Shape(1), MemoryMarshal.AsBytes<int>([1]).ToArray()));
                x = b.Add(b.Mul(x, b.Scale(register, -1f, 1f)), b.Mul(register, token));
            }

            // [rows, heads, 8] column blocks: queries (1, 0, ...), keys (bias, 0, ...); the group attention's keys in the group order.
            var unit = new float[p.HeadCount * BiasColumns];
            for (int h = 0; h < p.HeadCount; h++)
            {
                unit[h * BiasColumns] = 1f;
            }

            var first = b.Constant("bias_column", DataType.F32, new Shape(1, p.HeadCount, BiasColumns), MemoryMarshal.AsBytes<float>(unit).ToArray());
            queryColumns = b.Mul(b.Reshape(b.Scale(keyBias, 0f, 1f), 0, 1, 1), first);
            keyColumns = b.Mul(b.Reshape(keyBias, 0, 1, 1), first);
            groupKeyColumns = b.Mul(b.Reshape(b.Gather(keyBias, order), 0, 1, 1), first);
        }

        for (int layer = 0; layer < p.LayerCount; layer++)
        {
            using var _ = b.Group($"layers.{layer}");
            string n = $"layers.{layer}.";
            var h = b.RmsNorm(x, b.Weight(n + "time_attention_norm.weight"), p.NormEps);
            var rope = Attributes.From(("dims", (long)p.HeadDim), ("base", p.RopeBase), ("style", "neox"));
            var q = b.Emit1("Rope", [Heads(b, Linear(b, h, n + "time_attention.q.weight"), p), positions], rope);
            var k = b.Emit1("Rope", [Heads(b, Linear(b, h, n + "time_attention.k.weight"), p), positions], rope);
            var attended = Attend(b, q, k, Heads(b, Linear(b, h, n + "time_attention.v.weight"), p), queryColumns, keyColumns, spans);
            x = b.Add(x, Linear(b, b.Reshape(attended, 0, p.HeadCount * p.HeadDim), n + "time_attention.o.weight"));

            h = b.Gather(b.RmsNorm(x, b.Weight(n + "group_attention_norm.weight"), p.NormEps), order);
            q = Heads(b, Linear(b, h, n + "group_attention.q.weight"), p);
            k = Heads(b, Linear(b, h, n + "group_attention.k.weight"), p);
            attended = Attend(b, q, k, Heads(b, Linear(b, h, n + "group_attention.v.weight"), p), queryColumns, groupKeyColumns, groupSpans);
            x = b.Add(x, b.Gather(Linear(b, b.Reshape(attended, 0, p.HeadCount * p.HeadDim), n + "group_attention.o.weight"), inverse));

            h = b.RmsNorm(x, b.Weight(n + "mlp_norm.weight"), p.NormEps);
            x = b.Add(x, Linear(b, Activate(b, Linear(b, h, n + "mlp.up.weight"), p.Activation), n + "mlp.down.weight"));
        }

        using (b.Group("output"))
        {
            x = b.RmsNorm(b.Gather(x, read), b.Weight("output_norm.weight"), p.NormEps);
            b.Output(TimeSeriesGraphs.PatchQuantilesOutput, ResidualBlock(b, x, "output_patch_embedding", p.Activation));
        }

        return b.Build();
    }

    // Unscaled attention (Chronos-2 does not divide the scores by √d) with the key bias in the extra columns.
    private static Value Attend(GraphBuilder b, Value q, Value k, Value v, Value queryColumns, Value keyColumns, Value spans)
        => b.Emit1("Attention", [b.Concat(2, q, queryColumns), b.Concat(2, k, keyColumns), v, spans], Attributes.From(("scale", 1f), ("causal", false)));

    private static Value Heads(GraphBuilder b, Value x, Chronos2Parameters p) => b.Reshape(x, 0, p.HeadCount, p.HeadDim);

    // Chronos' ResidualBlock: output(act(hidden(x))) + residual(x).
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
        _ => throw new NotSupportedException($"chronos2: unknown activation '{activation}'."),
    };
}
