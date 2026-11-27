using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Compressed convolutional attention (CCA, ZAYA1) as a mixer of the <see cref="HybridAdapter"/>: the q/k projections are mixed over time by a
/// depthwise and then a grouped causal convolution (kernel 2 each), keep residual paths, are L2-normalized to sqrt(head_dim) with a learned key
/// temperature per key/value head, and the values carry a one-token delay.
/// </summary>
/// <remarks>
/// Weights below a layer prefix (C = (heads + kv_heads)·head_dim, G = heads + kv_heads groups of head_dim channels):
/// <list type="bullet">
/// <item><c>attention.{q,k,o}.weight</c> as for <see cref="HybridAttention"/>; <c>attention.v.weight</c> [kv_heads·head_dim, hidden] holds the
/// current-token half (first rows) and the delayed half of the values.</item>
/// <item><c>attention.cca.conv.weight</c> [C, 2] f32 and <c>attention.cca.conv.bias</c> [C]: the depthwise convolution of the concatenated
/// raw q/k rows.</item>
/// <item><c>attention.cca.group_conv.weight</c> [G, head_dim (out), 2·head_dim (in)]: per group the previous-row tap (first head_dim inputs) and
/// the current-row tap of the grouped convolution over the depthwise output; <c>attention.cca.group_conv.bias</c> [C].</item>
/// <item><c>attention.cca.key_temperature</c> [kv_heads]: the key scale after the L2 norm.</item>
/// </list>
/// Per token: <c>qk = [q, k]</c>, <c>d_t = conv(qk)_t</c> (taps qk_{t-1}, qk_t), <c>c_t = grouped(d_{t-1} + b, d_t + b) + b_g</c>; the residuals
/// <c>q_res = (q + k_{head/rep}) / 2</c> and <c>k_res</c> = the mean of q_res over the rep query heads of each key head; <c>q = c_q + q_res</c>,
/// <c>k = c_k + k_res</c>, both scaled to the norm sqrt(head_dim), keys times the temperature, then rope and attention as in
/// <see cref="HybridAttention.Attend"/>. Values: <c>[v_cur_t, v_delayed_{t-1}]</c> split into the kv heads. A sequence starts from zero rows
/// (the reference pads with zeros, so the first depthwise output before the sequence is the bias alone).
/// The three one-row windows (raw q/k, depthwise output without its bias, delayed value) are <c>CausalConv1d</c> states, so multi-sequence
/// batches, snapshots and rollback cover them like any convolution state.
/// </remarks>
internal sealed class CcaAttention
{
    private readonly HybridAttention _attention;

    /// <summary>Creates the mixer on top of the attention hyper-parameters of <paramref name="attention"/>.</summary>
    public CcaAttention(HybridAttention attention)
    {
        _attention = attention;
    }

    /// <summary>Whether the layer with prefix <paramref name="p"/> has a CCA mixer.</summary>
    public static bool Is(ModelSource src, string p) => src.Weight(p + "attention.cca.conv.weight") is not null;

    /// <summary>The recurrent windows of one layer: raw q/k rows, depthwise output, delayed values.</summary>
    public (StateDefinition QueryKey, StateDefinition Depthwise, StateDefinition Value) DefineStates(string group, int il)
    {
        int d = _attention.HeadDim;
        int channels = (_attention.HeadsOf(il) + _attention.KvHeadsOf(il)) * d;
        int delayed = _attention.KvHeadsOf(il) * d / 2;
        return (RecurrentStates.ConvWindow(group, "cca_qk_state", 1, channels), RecurrentStates.ConvWindow(group, "cca_conv_state", 1, channels),
            RecurrentStates.ConvWindow(group, "cca_v_state", 1, delayed));
    }

    /// <summary>Emits the mixer of layer <paramref name="il"/> on the normalized input <paramref name="h"/> [T, hidden].</summary>
    public Value Build(GraphBuilder b, Value h, Value positions, int il, string p, StateDefinition keyCache, StateDefinition valueCache,
        (StateDefinition QueryKey, StateDefinition Depthwise, StateDefinition Value) states)
    {
        int heads = _attention.HeadsOf(il);
        int kvHeads = _attention.KvHeadsOf(il);
        int d = _attention.HeadDim;
        int rep = heads / kvHeads;
        int groups = heads + kvHeads;
        int channels = groups * d;
        int half = kvHeads * d / 2;

        var q = b.Linear(h, p + "attention.q.weight");                                                        // [T, heads·d]
        var k = b.Linear(h, p + "attention.k.weight");                                                        // [T, kv·d]
        var v = b.Linear(h, p + "attention.v.weight");                                                        // [T, kv·d]

        // Residual paths: q_res = (q + k of its group) / 2, k_res = mean of q_res over the group.
        var qRes = b.Scale(b.Add(b.Reshape(q, 0, kvHeads, rep, d), b.Reshape(k, 0, kvHeads, 1, d)), 0.5f, 0f, $"{p}cca_q_res");  // [T, kv, rep, d]
        var kRes = b.Scale(b.Emit1("ReduceSum", [b.Permute(qRes, 0, 1, 3, 2)]), 1f / rep, 0f, $"{p}cca_k_res");     // [T, kv, d, 1]

        // Depthwise convolution (kernel 2) of the raw rows, then the grouped convolution over its previous and current rows.
        var qk = b.Concat(1, q, k);                                                                                // [T, C]
        var depthwise = Window(b, qk, b.Weight(p + "attention.cca.conv.weight"), states.QueryKey);                  // without its bias
        var previous = Window(b, depthwise, Selector(b, channels, $"{p}cca_shift"), states.Depthwise);
        var bias = b.Weight(p + "attention.cca.conv.bias");
        var taps = b.Concat(2, b.Reshape(b.Add(previous, bias), 0, groups, d), b.Reshape(b.Add(depthwise, bias), 0, groups, d));  // [T, G, 2d]
        var grouped = b.Emit1("MatMulId", [taps, b.Weight(p + "attention.cca.group_conv.weight"), GroupIds(b, qk, groups, p)], null, $"{p}cca_grouped");
        grouped = b.Add(grouped, b.Reshape(b.Weight(p + "attention.cca.group_conv.bias"), groups, d));               // [T, G, d]

        var query = b.Add(b.Reshape(b.Slice(grouped, 1, 0, heads), 0, kvHeads, rep, d), qRes);
        var key = b.Add(b.Reshape(b.Slice(grouped, 1, heads, kvHeads), 0, kvHeads, d, 1), kRes);
        float norm = MathF.Sqrt(d);
        query = b.Scale(b.L2Norm(b.Reshape(query, 0, heads, d), 1e-6f), norm, 0f, $"{p}cca_q");
        key = b.Scale(b.L2Norm(b.Reshape(key, 0, kvHeads, d), 1e-6f), norm);
        key = b.Mul(key, b.Reshape(b.Weight(p + "attention.cca.key_temperature"), kvHeads, 1), $"{p}cca_k");

        // Values: the current half of this row and the delayed half of the previous row.
        var delayed = Window(b, b.Slice(v, 1, half, half), Selector(b, half, $"{p}cca_v_shift"), states.Value);
        var value = b.Concat(1, b.Slice(v, 1, 0, half), delayed);                                                  // [T, kv·d]

        return _attention.Attend(b, h, b.Reshape(query, 0, heads * d), b.Reshape(key, 0, kvHeads * d), value, positions, il, p, keyCache, valueCache);
    }

    // Causal two-tap convolution of the rows with the kernel [C, 2] over a one-row window state.
    private static Value Window(GraphBuilder b, Value x, Value kernel, StateDefinition window)
    {
        var state = b.State(window);
        return b.Emit("CausalConv1d", [x, kernel, state], Attributes.From(("activation", "none")), writes: [state])[0];
    }

    // The kernel (1, 0) per channel: row t of the convolution is row t-1 of its input.
    private static Value Selector(GraphBuilder b, int channels, string name)
    {
        var bytes = new byte[channels * 2 * sizeof(float)];
        for (int c = 0; c < channels; c++)
        {
            BitConverter.TryWriteBytes(bytes.AsSpan(c * 2 * sizeof(float), sizeof(float)), 1f);
        }

        return b.Constant(name, DataType.F32, new Shape(channels, 2), bytes);
    }

    // The group index of every slot, i32 [T, G] (0 … G − 1 in each row; the row count follows the batch).
    private static Value GroupIds(GraphBuilder b, Value rows, int groups, string p)
    {
        var iotaBytes = new byte[groups * 4];
        for (int g = 0; g < groups; g++)
        {
            BitConverter.TryWriteBytes(iotaBytes.AsSpan(g * 4, 4), (float)g);
        }

        var zeros = b.Scale(b.Slice(rows, 1, 0, 1), 0f);
        var iota = b.Constant($"{p}cca_group_iota", DataType.F32, new Shape(groups), iotaBytes);
        return b.Emit1("Cast", [b.Add(zeros, iota)], Attributes.From(("type", "i32")), $"{p}cca_group_ids");
    }
}
