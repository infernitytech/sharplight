using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Hashed n-gram memory read by some layers of a hyper-connected decoder (DeepSeek-V4.1's Engram): every token and its predecessors, mapped to
/// hashed ids, are hashed per layer module into rows of one large table (<see cref="EngramOptions"/>, op <c>NGramHash</c>); the rows of a
/// token give one key per residual stream and one value shared by all of them, and every stream adds the value gated by how well its key
/// matches the stream.
/// </summary>
/// <remarks>
/// Weights: the table <c>engram.table.weight</c> [rows, Dh] (any gatherable format) and below the layer prefix <c>engram.kv.weight</c>
/// [(N + 1)·H, heads·Dh] and <c>engram.gate</c> [N, H] (f32), for N streams of H channels. With e the concatenated rows of the token's hash
/// heads, [k_0 .. k_{N-1} | v] = kv · e and per stream s: g = Σ_d rms(X_s)[d]·rms(k_s)[d]·gate[s, d] / sqrt(H),
/// X_s += sigmoid(sign(g)·sqrt(max(|g|, 1e-6)))·v (the norms without weights). A token without a hashed id (and every predecessor before a
/// missing one, or before the sequence start) hashes as the pad id.
/// Hunyuan v4 Flash adds optional weights below the layer prefix: <c>engram.lookup_norm.weight</c> [heads, Dh] (every looked-up row is RMS-normed
/// and scaled by its head's weight), <c>engram.kv.bias</c> [N + 1, H], <c>engram.pre_gamma</c> [H] (the value is RMS-normed and scaled by it)
/// and <c>engram.hc_scale</c> [N, H] (stream s adds v·(gate + hc_scale[s]) instead of v·gate). The token window comes from a causal convolution used as a shift register
/// over (hashed id + 1), carried per sequence like any recurrent state (a cleared state reads as "no token").
/// </remarks>
internal static class EngramMemory
{
    private const float GateFloor = 1e-6f;
    private const int MapWidth = 8;

    /// <summary>The token-window state [N - 1, N] shared by every module.</summary>
    public static StateDefinition WindowState(EngramOptions options) => new()
    {
        Name = "engram.window", Type = DataType.F32, Shape = new Shape(options.Order - 1, options.Order), Kind = StateKind.Recurrent, Group = "engram",
    };

    /// <summary>The window [T, N] of hashed ids + 1 (column j: the token j positions back; 0 = none) of every row.</summary>
    public static Value Window(GraphBuilder b, Value tokens, EngramOptions options, Value state)
    {
        int order = options.Order;
        var shift = new float[order * order];
        for (int j = 0; j < order; j++)
        {
            shift[j * order + order - 1 - j] = 1f;
        }

        // Rows of MapWidth values (the id in column 0) keep the table gatherable by every device.
        var ids = new float[options.TokenMap.Length * MapWidth];
        for (int t = 0; t < options.TokenMap.Length; t++)
        {
            ids[t * MapWidth] = options.TokenMap[t] < 0 ? 0f : options.TokenMap[t] + 1f;
        }

        var map = b.Constant("engram.token_map", DataType.F32, new Shape(options.TokenMap.Length, MapWidth), MemoryMarshal.AsBytes(ids.AsSpan()).ToArray());
        var current = b.Slice(b.Gather(map, tokens), 1, 0, 1, "engram_ids");                                         // [T, 1]
        var tiled = b.Emit1("Tile", [current], Attributes.From(("repeats", new long[] { 1, order })));
        return b.Emit("CausalConv1d",
            [tiled, b.Constant("engram.shift", DataType.F32, new Shape(order, order), MemoryMarshal.AsBytes(shift.AsSpan()).ToArray()), state],
            Attributes.From(("activation", "none")), writes: [state])[0];
    }

    /// <summary>The streams x [T, N·H] after the module of layer prefix <paramref name="p"/> (module index <paramref name="module"/>) added its memory.</summary>
    public static Value Build(GraphBuilder b, Value window, Value x, string p, int module, EngramOptions options, int streams, int hidden, float eps)
    {
        eps = options.NormEps ?? eps;
        int heads = options.Heads;
        var ids = b.Emit1("NGramHash", [window], Attributes.From(
            ("multipliers", options.Multipliers[(module * options.Order)..((module + 1) * options.Order)]),
            ("head_sizes", options.HeadSizes[(module * heads)..((module + 1) * heads)]),
            ("head_offsets", options.HeadOffsets[(module * heads)..((module + 1) * heads)]),
            ("heads_per_order", (long)options.HeadsPerOrder), ("stop_token", options.Pad), ("stop_cuts", false)), $"{p}engram_hash");

        var table = b.Weight("engram.table.weight");
        var rows = Enumerable.Range(0, heads).Select(h => b.Gather(table, b.Reshape(b.Slice(ids, 1, h, 1), 0))).ToArray();
        var embedding = rows.Length == 1 ? rows[0] : b.Concat(1, rows);
        if (b.TryWeight(p + "engram.lookup_norm.weight") is { } lookupNorm)
        {
            long dims = lookupNorm.Shape[1].Fixed;
            embedding = b.Reshape(b.Mul(b.RmsNorm(b.Reshape(embedding, 0, heads, dims), null, eps), lookupNorm), 0, heads * dims);
        }

        var kv = b.Linear(embedding, p + "engram.kv.weight", name: $"{p}engram_kv");                                     // [T, (N + 1)·H]
        if (b.TryWeight(p + "engram.kv.bias") is { } bias)
        {
            kv = b.Add(kv, b.Reshape(bias, (long)(streams + 1) * hidden));   // stored [N + 1, H] or flat
        }
        var key = b.RmsNorm(b.Reshape(b.Slice(kv, 1, 0, (long)streams * hidden), 0, streams, hidden), null, eps);
        var value = b.Reshape(b.Slice(kv, 1, (long)streams * hidden, hidden), 0, 1, hidden);
        var query = b.RmsNorm(b.Reshape(x, 0, streams, hidden), null, eps);
        var match = b.Mul(b.Mul(query, key), b.Weight(p + "engram.gate"));
        var g = b.Scale(b.Emit1("ReduceSum", [match]), 1f / MathF.Sqrt(hidden));                                       // [T, N, 1]
        var floor = b.Constant(p + "engram.gate_floor", DataType.F32, new Shape(1), BitConverter.GetBytes(GateFloor));
        var gate = b.Sigmoid(b.Mul(b.Unary("Sign", g), b.Unary("Sqrt", b.Binary("Max", b.Unary("Abs", g), floor))));
        if (b.TryWeight(p + "engram.pre_gamma") is { } gamma)
        {
            value = b.Mul(b.RmsNorm(value, null, eps), gamma);
        }

        var scale = b.TryWeight(p + "engram.hc_scale") is { } offsets ? b.Add(gate, offsets) : gate;                     // [T, N, 1] or [T, N, H]
        var added = b.Add(b.Reshape(x, 0, streams, hidden), b.Mul(scale, value), $"{p}engram_out");
        return b.Reshape(added, 0, (long)streams * hidden);
    }
}
