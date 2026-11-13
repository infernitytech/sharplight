using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Hashed n-gram embedding injected into the hyper-connection residual streams of one layer (per-layer embedding, "PLE").
/// Every token's preceding tokens are hashed into rows of a large table (<see cref="NGramOptions"/>, op <c>NGramHash</c>); the
/// concatenated rows give one value shared by all streams and one key per stream. The key, matched against the normalized
/// stream, gates the value; a dilated depthwise causal convolution of the normalized gated value adds local context.
/// </summary>
/// <remarks>
/// Weights below <c>layers.{i}.ngram.</c>: <c>table.weight</c> [rows, D] (any gatherable format), <c>key.weight</c>
/// [streams·H, heads·D], <c>value.weight</c> [H, heads·D], <c>key_norm.weight</c>, <c>query_norm.weight</c>,
/// <c>conv_norm.weight</c> [streams·H] (per-stream RMS norms, multiplied directly) and <c>conv1d.weight</c> [streams·H, K] f32
/// (dilation <see cref="NGramOptions.ConvDilation"/>). With e the embedding, q = norm(X), k = norm(key(e)):
/// g = (k·q) / sqrt(H) per stream, gate = sigmoid(sign(g)·sqrt(max(|g|, 1e-6))), v = gate · value(e);
/// output v + silu(conv(norm(v))), added to the streams before the layer's attention mixer.
/// The token window comes from a causal convolution used as a shift register over (token + 1), so it is carried per sequence
/// like any other recurrent state and a cleared state reads as "no token".
/// </remarks>
internal static class NGramEmbedding
{
    private const float GateFloor = 1e-6f;

    /// <summary>Name of the dense (dilation folded in) convolution weight the adapter derives for layer prefix <paramref name="p"/>.</summary>
    private static string DenseConvName(string p) => p + "ngram.conv1d.dense";

    /// <summary>The token-window state [order-1, order] and the convolution state [(K-1)·dilation, channels] of one layer.</summary>
    public static (StateDefinition Window, StateDefinition Conv) States(string group, NGramOptions options, WeightInfo conv)
    {
        long channels = conv.Shape[0].Fixed;
        long span = DenseKernel(conv, options.ConvDilation) - 1;
        return (
            new StateDefinition { Name = $"{group}.ngram_window", Type = DataType.F32, Shape = new Shape(options.Order - 1, options.Order), Kind = StateKind.Recurrent, Group = group },
            new StateDefinition { Name = $"{group}.ngram_conv_state", Type = DataType.F32, Shape = new Shape(span, channels), Kind = StateKind.Recurrent, Group = group });
    }

    /// <summary>Adds the convolution weight with the dilation folded in (zero taps between the real ones).</summary>
    public static void AddDerivedWeights(IDictionary<string, WeightInfo> weights, NGramOptions options)
    {
        string p = $"layers.{options.Layer}.";
        var conv = weights[p + "ngram.conv1d.weight"];
        if (conv.Type != DataType.F32)
        {
            throw new InvalidDataException($"{conv.Name} must be f32.");
        }

        int dilation = options.ConvDilation;
        long channels = conv.Shape[0].Fixed;
        long kernel = conv.Shape[1].Fixed;
        long dense = DenseKernel(conv, dilation);
        weights[DenseConvName(p)] = WeightInfo.Derived(DenseConvName(p), DataType.F32, new Shape(channels, dense), () =>
        {
            var src = MemoryMarshal.Cast<byte, float>(conv.ReadAll());
            var dst = new float[channels * dense];
            for (long c = 0; c < channels; c++)
            {
                for (long k = 0; k < kernel; k++)
                {
                    dst[c * dense + k * dilation] = src[(int)(c * kernel + k)];
                }
            }

            return MemoryMarshal.AsBytes(dst.AsSpan()).ToArray();
        });
    }

    private static long DenseKernel(WeightInfo conv, int dilation) => (conv.Shape[1].Fixed - 1) * dilation + 1;

    /// <summary>The n-gram features [T, streams·H] to add to the residual streams x of the layer with prefix <paramref name="p"/>.</summary>
    public static Value Build(GraphBuilder b, Value tokens, Value x, string p, NGramOptions options, int streams, int hidden, float eps,
        Value windowState, Value convState)
    {
        int order = options.Order;
        // Shift register: channel j of the window is the input j tokens back; inputs are token + 1 so a zero state means "none".
        var shift = new float[order * order];
        for (int j = 0; j < order; j++)
        {
            shift[j * order + order - 1 - j] = 1f;
        }

        var current = b.Scale(b.Reshape(b.Emit1("Cast", [tokens], Attributes.From(("type", DataType.F32.Name))), 0, 1), 1f, 1f);
        var tiled = b.Emit1("Tile", [current], Attributes.From(("repeats", new long[] { 1, order })));
        var window = b.Emit("CausalConv1d",
            [tiled, b.Constant(p + "ngram.shift", DataType.F32, new Shape(order, order), MemoryMarshal.AsBytes(shift.AsSpan()).ToArray()), windowState],
            Attributes.From(("activation", "none")), writes: [windowState])[0];
        var ids = b.Emit1("NGramHash", [window], Attributes.From(
            ("multipliers", options.Multipliers), ("head_sizes", options.HeadSizes), ("head_offsets", options.HeadOffsets),
            ("heads_per_order", (long)options.HeadsPerOrder), ("stop_token", options.StopToken)), $"{p}ngram_ids");

        var table = b.Weight(p + "ngram.table.weight");
        var rows = Enumerable.Range(0, options.HeadSizes.Length)
            .Select(h => b.Gather(table, b.Reshape(b.Slice(ids, 1, h, 1), 0)))
            .ToArray();
        var embedding = rows.Length == 1 ? rows[0] : b.Concat(1, rows);

        var key = HyperConnections.StreamNorm(b, b.Linear(embedding, p + "ngram.key.weight"), b.Weight(p + "ngram.key_norm.weight"), streams, hidden, eps);
        var query = HyperConnections.StreamNorm(b, x, b.Weight(p + "ngram.query_norm.weight"), streams, hidden, eps);
        var value = b.Linear(embedding, p + "ngram.value.weight");
        var g = b.Scale(b.Emit1("ReduceSum", [b.Mul(key, query)]), 1f / MathF.Sqrt(hidden));
        var floor = b.Constant(p + "ngram.gate_floor", DataType.F32, new Shape(1), BitConverter.GetBytes(GateFloor));
        var gate = b.Sigmoid(b.Mul(b.Unary("Sign", g), b.Unary("Sqrt", b.Binary("Max", b.Unary("Abs", g), floor))));
        var gated = b.Reshape(b.Mul(gate, b.Reshape(value, 0, 1, hidden)), 0, (long)streams * hidden);

        var normed = b.Reshape(HyperConnections.StreamNorm(b, gated, b.Weight(p + "ngram.conv_norm.weight"), streams, hidden, eps), 0, (long)streams * hidden);
        var conv = b.Emit("CausalConv1d", [normed, b.Weight(DenseConvName(p)), convState], Attributes.From(("activation", "silu")), writes: [convState])[0];
        return b.Add(gated, conv, $"{p}ngram_out");
    }
}
