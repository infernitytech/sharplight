using Sharplight.Graph;

namespace Sharplight.Architectures;

/// <summary>
/// A per-token memory across depth: entries written at chosen points of a layer stack (a key and a value of width M each) and read back by
/// later layers of the same token with softmax attention from every residual stream. Nothing is carried across tokens; one instance follows
/// one graph's walk through the layers.
/// </summary>
/// <remarks>
/// Weights below <c>depth_memory.</c>: <c>query.weight</c> [M, H] / <c>query.bias</c> [M], <c>key.weight</c>, <c>value.weight</c> [M, M] with
/// biases [M], <c>out.weight</c> [H, M] / <c>out.bias</c> [H], <c>stream_gate.weight</c> [S, H] / <c>stream_gate.bias</c> [S] (a sigmoid gate per
/// stream on the retrieved projection), <c>write_state.weight</c> [M, H] (the entry from the stream mean), <c>write_meta.weight</c> [M, N] (the
/// entry from N caller-chosen signals) and <c>entry_bias</c> [entries, M] (a learned offset per entry index). The routing features of
/// <see cref="RouteFeatures"/> use <c>route.{state,retrieved,diff}.weight</c> [R, H], <c>route.history.weight</c> [R, R] and <c>route.bias</c> [R].
/// </remarks>
internal sealed class DepthMemory
{
    private const string Prefix = "depth_memory.";
    private readonly List<Value> _keys = [];
    private readonly List<Value> _values = [];
    private readonly int _streams;
    private readonly int _hidden;
    private readonly int _width;

    /// <summary>Starts an empty memory for streams [T, S, H] and entries of width <paramref name="width"/>.</summary>
    public DepthMemory(int streams, int hidden, int width)
    {
        _streams = streams;
        _hidden = hidden;
        _width = width;
    }

    /// <summary>Whether any entry was written.</summary>
    public bool IsEmpty => _keys.Count == 0;

    /// <summary>Number of entries written so far (the index of the next one).</summary>
    public int Count => _keys.Count;

    /// <summary>
    /// Every stream of <paramref name="x"/> [T, S, H] attends over the entries; the projected result is added to the stream through its gate.
    /// </summary>
    /// <returns>The updated streams and the stream mean of the projected results [T, H].</returns>
    public (Value Streams, Value Retrieved) Read(GraphBuilder b, Value x)
    {
        var keys = _keys.Count == 1 ? _keys[0] : b.Concat(1, [.. _keys]);                                                      // [T, e, M]
        var values = _values.Count == 1 ? _values[0] : b.Concat(1, [.. _values]);
        var q = b.Linear(x, Prefix + "query.weight", Prefix + "query.bias");                                                   // [T, S, M]
        var scores = b.Emit1("BatchMatMul", [q, keys], Attributes.From(("trans_b", true), ("alpha", 1f / MathF.Sqrt(_width))));   // [T, S, e]
        var retrieved = b.Emit1("BatchMatMul", [b.Emit1("Softmax", [scores]), values]);                                        // [T, S, M]
        var projected = b.Linear(retrieved, Prefix + "out.weight", Prefix + "out.bias", $"depth_memory_read_{_keys.Count}");    // [T, S, H]
        var gate = b.Emit1("ReduceSum", [b.Mul(x, b.Weight(Prefix + "stream_gate.weight"))]);                                   // [T, S, 1]
        gate = b.Sigmoid(b.Add(gate, b.Reshape(b.Weight(Prefix + "stream_gate.bias"), _streams, 1)));
        return (b.Add(x, b.Mul(projected, gate)), StaticStreamMixing.Mean(b, projected, _streams, _hidden));
    }

    /// <summary>Appends an entry made from the stream mean <paramref name="state"/> [T, H] and the signals <paramref name="meta"/> [T, N].</summary>
    public void Write(GraphBuilder b, Value state, Value meta)
    {
        var bias = b.Slice(b.Weight(Prefix + "entry_bias"), 0, _keys.Count, 1);                                                // [1, M]
        var entry = b.Add(b.Add(b.Linear(state, Prefix + "write_state.weight"), b.Linear(meta, Prefix + "write_meta.weight")), bias);
        _keys.Add(b.Reshape(b.Linear(entry, Prefix + "key.weight", Prefix + "key.bias"), 0, 1, _width));
        _values.Add(b.Reshape(b.Linear(entry, Prefix + "value.weight", Prefix + "value.bias"), 0, 1, _width));
    }

    /// <summary>
    /// Routing features tanh(W_s·state + W_r·retrieved + W_d·diff + W_h·history + b) [T, R]; the optional terms are left out when null.
    /// </summary>
    public static Value RouteFeatures(GraphBuilder b, Value state, Value? retrieved, Value? diff, Value? history)
    {
        var f = b.Linear(state, Prefix + "route.state.weight");
        if (retrieved is not null)
        {
            f = b.Add(f, b.Linear(retrieved, Prefix + "route.retrieved.weight"));
        }

        if (diff is not null)
        {
            f = b.Add(f, b.Linear(diff, Prefix + "route.diff.weight"));
        }

        if (history is not null)
        {
            f = b.Add(f, b.Linear(history, Prefix + "route.history.weight"));
        }

        return b.Unary("Tanh", b.Add(f, b.Weight(Prefix + "route.bias")));
    }
}
