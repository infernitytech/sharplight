using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;
using Sharplight.TimeSeries;

namespace Sharplight.Architectures;

/// <summary>
/// Emits the Moirai forecasters (see <see cref="TimeSeriesGraphs"/>): uni2ts' pre-norm transformer (RMS norms, query / key RMS norms per head, rotary time
/// positions on the first half of every head with GPT-J pairs, a SiLU-gated feed-forward block, a final RMS norm) between
/// <list type="bullet">
/// <item><description>Moirai 1.x (<see cref="TimeSeriesGraphs.MaskedEncoderGraph"/>): the linear patch projection of one patch size, the learned mask embedding
/// on the horizon rows, bidirectional attention, and the fused distribution head of that patch size on the read rows;</description></item>
/// <item><description>Moirai 2.0 (<see cref="TimeSeriesGraphs.QuantileDecoderGraph"/>): residual-block input and output projections around attention that
/// the host's spans make causal in time.</description></item>
/// </list>
/// The variate attention bias is not emitted: within one univariate series it adds the same value to every score of a head.
/// </summary>
internal static class MoiraiGraph
{
    /// <summary>Builds the masked-encoder graph of patch size <paramref name="patchSize"/> (Moirai 1.x).</summary>
    public static ComputeGraph BuildEncoder(MoiraiParameters p, int patchSize, Func<string, WeightInfo?> weights)
    {
        var b = new GraphBuilder(TimeSeriesGraphs.MaskedEncoderGraph(patchSize), weights);
        var rows = Dim.Sym(TimeSeriesGraphs.PatchSymbol);
        var patches = b.Input(TimeSeriesGraphs.PatchesInput, InputRole.Other, DataType.F32, new Shape(rows, patchSize));
        var predict = b.Input(TimeSeriesGraphs.PredictMaskInput, InputRole.Other, DataType.F32, new Shape(rows, 1));
        var (positions, spans, read) = Common(b);
        Value x;
        using (b.Group("input"))
        {
            x = Linear(b, patches, $"in_proj.p{patchSize}");
            x = b.Add(b.Mul(x, b.Scale(predict, -1f, 1f)), b.Mul(predict, b.Weight("mask_encoding.weight")));
        }

        x = Encoder(b, x, positions, spans, p);
        using (b.Group("output"))
        {
            b.Output(TimeSeriesGraphs.DistributionOutput, Linear(b, b.Gather(x, read), $"param_proj.p{patchSize}"));
        }

        return b.Build();
    }

    /// <summary>Builds the quantile-decoder graph (Moirai 2.0).</summary>
    public static ComputeGraph BuildDecoder(MoiraiParameters p, Func<string, WeightInfo?> weights)
    {
        var b = new GraphBuilder(TimeSeriesGraphs.QuantileDecoderGraph, weights);
        var rows = Dim.Sym(TimeSeriesGraphs.PatchSymbol);
        var patches = b.Input(TimeSeriesGraphs.PatchesInput, InputRole.Other, DataType.F32, new Shape(rows, 2 * p.PatchSize));
        var (positions, spans, read) = Common(b);
        Value x;
        using (b.Group("input"))
        {
            x = ResidualBlock(b, patches, "in_proj");
        }

        x = Encoder(b, x, positions, spans, p);
        using (b.Group("output"))
        {
            b.Output(TimeSeriesGraphs.TokenQuantilesOutput, ResidualBlock(b, b.Gather(x, read), "out_proj"));
        }

        return b.Build();
    }

    private static (Value Positions, Value Spans, Value Read) Common(GraphBuilder b)
    {
        var rows = Dim.Sym(TimeSeriesGraphs.PatchSymbol);
        var positions = b.Input(TimeSeriesGraphs.PositionsInput, InputRole.Positions, DataType.I32, new Shape(rows, 1));
        var spans = b.Input(TimeSeriesGraphs.SpansInput, InputRole.RowSpans, DataType.I32, new Shape(rows, 2));
        var read = b.Input(TimeSeriesGraphs.ReadRowsInput, InputRole.Other, DataType.I32, new Shape(Dim.Sym(TimeSeriesGraphs.ReadSymbol)));
        return (positions, spans, read);
    }

    private static Value Encoder(GraphBuilder b, Value x, Value positions, Value spans, MoiraiParameters p)
    {
        for (int layer = 0; layer < p.LayerCount; layer++)
        {
            using var _ = b.Group($"layers.{layer}");
            x = Layer(b, x, $"encoder.layers.{layer}.", positions, spans, p);
        }

        using (b.Group("norm"))
        {
            return b.RmsNorm(x, b.Weight("encoder.norm.weight"), p.NormEps);
        }
    }

    // x + out(attention(rms(x))), then x + fc2(silu(gate(rms(x))) · fc1(rms(x))).
    private static Value Layer(GraphBuilder b, Value x, string n, Value positions, Value spans, MoiraiParameters p)
    {
        var h = b.RmsNorm(x, b.Weight(n + "norm1.weight"), p.NormEps);
        string a = n + "self_attn.";
        var rope = Attributes.From(("dims", (long)(p.HeadDim / 2)), ("base", p.RopeBase), ("style", "gptj"));
        var q = b.Emit1("Rope", [b.RmsNorm(Heads(b, Linear(b, h, a + "q_proj"), p), b.Weight(a + "q_norm.weight"), p.NormEps), positions], rope);
        var k = b.Emit1("Rope", [b.RmsNorm(Heads(b, Linear(b, h, a + "k_proj"), p), b.Weight(a + "k_norm.weight"), p.NormEps), positions], rope);
        var v = Heads(b, Linear(b, h, a + "v_proj"), p);
        var attended = b.Emit1("Attention", [q, k, v, spans], Attributes.From(("scale", 1f / MathF.Sqrt(p.HeadDim)), ("causal", false)));
        x = b.Add(x, Linear(b, b.Reshape(attended, 0, p.HiddenSize), a + "out_proj"));
        var f = b.RmsNorm(x, b.Weight(n + "norm2.weight"), p.NormEps);
        var gated = b.Mul(b.Silu(Linear(b, f, n + "ffn.fc_gate")), Linear(b, f, n + "ffn.fc1"));
        return b.Add(x, Linear(b, gated, n + "ffn.fc2"));
    }

    private static Value Heads(GraphBuilder b, Value x, MoiraiParameters p) => b.Reshape(x, 0, p.HeadCount, p.HeadDim);

    // uni2ts' ResidualBlock: output(silu(hidden(x))) + residual(x).
    private static Value ResidualBlock(GraphBuilder b, Value x, string module)
        => b.Add(Linear(b, b.Silu(Linear(b, x, module + ".hidden_layer")), module + ".output_layer"), Linear(b, x, module + ".residual_layer"));

    private static Value Linear(GraphBuilder b, Value x, string module)
    {
        var y = b.Emit1("MatMul", [x, b.Weight(module + ".weight")], Attributes.From((StandardOps.F32ActivationsAttribute, true)));
        return b.TryWeight(module + ".bias") is { } bias ? b.Add(y, bias) : y;
    }
}
