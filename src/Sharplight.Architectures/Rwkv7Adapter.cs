using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// RWKV-7 "Goose" (canonical arch <c>rwkv7</c>): attention-free recurrent language model. Every layer is a time-mix block (token-shift interpolations, LoRA-parametrised
/// decay, in-context learning rate, value residual and output gate feeding the <c>Wkv7</c> generalised delta-rule recurrence, per-head group norm) followed by a
/// channel-mix block (token-shift, squared-ReLU MLP). All norms are LayerNorms with bias.
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>rwkv7.*</c>): <c>layer_count</c>, <c>hidden_size</c>, <c>head_size</c> (N; heads = hidden / N), <c>norm_eps</c>, <c>context_length</c>.
/// Weights: <c>token_embedding.weight</c> [V, C], <c>embedding_norm.{weight,bias}</c>, <c>output_norm.{weight,bias}</c>, <c>lm_head.weight</c>; per layer <c>input_norm.{weight,bias}</c>,
/// <c>post_attention_norm.{weight,bias}</c>, and below <c>time_mix.</c>: <c>lerp.weight</c> [6, C] (interpolation factors for r, w, k, v, a, g), <c>{receptance,key,value,output}.weight</c> [C, C],
/// <c>decay_bias</c> [C] (w0), <c>decay_down.weight</c> [Dw, C], <c>decay_up.weight</c> [C, Dw], <c>iclr_bias</c>, <c>iclr_down.weight</c>, <c>iclr_up.weight</c> (in-context learning rate),
/// <c>value_mix_bias</c>, <c>value_mix_down.weight</c>, <c>value_mix_up.weight</c> (value residual, layers after the first), optional <c>gate_down.weight</c> [Dg, C], <c>gate_up.weight</c> [C, Dg],
/// <c>key_norm_scale</c> [C] (k_k), <c>key_lr_scale</c> [C] (k_a), <c>bonus</c> [C] (r_k), <c>out_norm.{weight,bias}</c> [C] (group norm affine);
/// below <c>channel_mix.</c>: <c>lerp.weight</c> [C], <c>key.weight</c> [I, C], <c>value.weight</c> [C, I].
/// Per-layer states: the last time-mix and channel-mix inputs (token shift) and the [H, N, N] wkv state.
/// </remarks>
public sealed class Rwkv7Adapter : IModelAdapter
{
    private const float GroupNormEps = 64e-5f;
    private const float DecayScale = -0.606531f;

    /// <inheritdoc/>
    public string Name => "RWKV-7";
    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => ["rwkv7"];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        int layerCount = src.Arch<int>("layer_count");
        int hidden = src.Arch<int>("hidden_size");
        int headSize = src.Arch("head_size", 64);
        int heads = hidden / headSize;
        float eps = src.Arch("norm_eps", 1e-5f);
        var shift = new RwkvTokenShift(hidden);

        var states = new List<StateDefinition>();
        var timeShift = new StateDefinition[layerCount];
        var channelShift = new StateDefinition[layerCount];
        var wkvState = new StateDefinition[layerCount];
        for (int il = 0; il < layerCount; il++)
        {
            string g = $"layers.{il}";
            timeShift[il] = shift.DefineState($"{g}.time_shift", g);
            channelShift[il] = shift.DefineState($"{g}.channel_shift", g);
            wkvState[il] = new StateDefinition { Name = $"{g}.wkv_state", Type = DataType.F32, Shape = new Shape(heads, headSize, headSize), Kind = StateKind.Recurrent, Group = g };
            states.AddRange([timeShift[il], channelShift[il], wkvState[il]]);
        }

        var graphs = new Dictionary<string, ComputeGraph> { [WellKnown.DecodeGraph] = BuildTrunk(WellKnown.DecodeGraph, false) };
        if (options.IncludeAuxiliaryGraphs)
        {
            graphs["decode_embeddings"] = BuildTrunk("decode_embeddings", true);
        }

        return RecurrentModelDefinition.Create(src, layerCount, hidden, graphs, states);

        ComputeGraph BuildTrunk(string name, bool embeddingsInput)
        {
            var b = new GraphBuilder(name, src.Weight);
            var T = Dim.Sym(WellKnown.TokenSymbol);
            Value x;
            using (b.Group("input"))
            {
                x = embeddingsInput
                    ? b.Input("embeddings", InputRole.Embeddings, DataType.F32, new Shape(T, hidden))
                    : b.Gather(b.Weight("token_embedding.weight"), b.Input("tokens", InputRole.TokenIds, DataType.I32, new Shape(T)), "inp_embd");
                x = LayerNorm(b, x, "embedding_norm");
            }

            b.Input("positions", InputRole.Positions, DataType.I32, new Shape(T, 1));
            var outRows = b.Input("output_rows", InputRole.OutputRows, DataType.I32, new Shape(Dim.Sym("R")));
            var selector = shift.Selector(b);
            Value? firstValue = null;
            for (int il = 0; il < layerCount; il++)
            {
                using var _ = b.Group($"layers.{il}");
                string p = $"layers.{il}.";
                var h = LayerNorm(b, x, p + "input_norm");
                var (attention, value) = TimeMix(b, h, p, il, selector, firstValue);
                firstValue ??= value;
                x = b.Add(attention, x, $"{p}attn_out");
                x = b.Add(ChannelMix(b, LayerNorm(b, x, p + "post_attention_norm"), p, il, selector), x, $"{p}l_out");
            }

            using (b.Group("output"))
            {
                var hiddenStates = LayerNorm(b, x, "output_norm");
                var rows = b.Gather(hiddenStates, outRows, "selected_rows");
                var head = b.TryWeight("lm_head.weight") ?? b.Weight("token_embedding.weight");
                b.Output("logits", b.MatMul(rows, head, "logits"));
                b.Output("hidden", hiddenStates);
            }

            return b.Build();
        }

        Value LayerNorm(GraphBuilder b, Value v, string prefix)
            => b.Emit1("LayerNorm", [v, b.Weight(prefix + ".weight"), b.Weight(prefix + ".bias")], Attributes.From(("eps", eps)));

        // Returns the time-mix output and this layer's value projection (layer 0's feeds the value residual of later layers).
        (Value Output, Value Value) TimeMix(GraphBuilder b, Value h, string p, int il, Value selector, Value? firstValue)
        {
            string t = p + "time_mix.";
            var previous = shift.Previous(b, h, selector, timeShift[il]);
            var delta = b.Sub(previous, h);
            var lerp = b.Weight(t + "lerp.weight");                                        // [6, C]
            Value Mixed(int index) => b.Add(h, b.Mul(delta, b.Reshape(b.Slice(lerp, 0, index, 1), hidden)));
            var xr = Mixed(0);
            var xw = Mixed(1);
            var xk = Mixed(2);
            var xv = Mixed(3);
            var xa = Mixed(4);
            var xg = Mixed(5);

            var r = b.Linear(xr, t + "receptance.weight");
            var decayLog = b.Add(b.Weight(t + "decay_bias"), b.Linear(b.Unary("Tanh", b.Linear(xw, t + "decay_down.weight")), t + "decay_up.weight"));
            var w = b.Unary("Exp", b.Scale(b.Sigmoid(decayLog), DecayScale));
            var k = b.Linear(xk, t + "key.weight");
            var v = b.Linear(xv, t + "value.weight");
            var projectedValue = v;
            if (firstValue is not null)
            {
                var mix = b.Sigmoid(b.Add(b.Weight(t + "value_mix_bias"), b.Linear(b.Linear(xv, t + "value_mix_down.weight"), t + "value_mix_up.weight")));
                v = b.Add(v, b.Mul(b.Sub(firstValue, v), mix));
            }

            var a = b.Sigmoid(b.Add(b.Weight(t + "iclr_bias"), b.Linear(b.Linear(xa, t + "iclr_down.weight"), t + "iclr_up.weight")));
            var kk = b.Reshape(b.L2Norm(b.Reshape(b.Mul(k, b.Weight(t + "key_norm_scale")), 0, heads, headSize), 1e-12f), 0, hidden);
            k = b.Mul(k, b.Scale(b.Mul(b.Scale(a, 1f, -1f), b.Weight(t + "key_lr_scale")), 1f, 1f));                 // k · (1 + (a - 1) · k_a)

            Value Heads(Value z) => b.Reshape(z, 0, heads, headSize);
            var ws = b.State(wkvState[il]);
            var wkv = b.Emit("Wkv7", [Heads(r), Heads(w), Heads(k), Heads(v), Heads(b.Scale(kk, -1f)), Heads(b.Mul(kk, a)), ws], null, $"{p}wkv", writes: [ws])[0];

            // Per-head group norm of the recurrence output, plus the "bonus" of the current token: (Σⱼ r·k·r_k) · v per head.
            var normed = b.Emit1("LayerNorm", [wkv], Attributes.From(("eps", GroupNormEps)));                      // [T, H, N], statistics per head
            var y = b.Add(b.Mul(b.Reshape(normed, 0, hidden), b.Weight(t + "out_norm.weight")), b.Weight(t + "out_norm.bias"));
            var bonus = b.Emit1("ReduceSum", [Heads(b.Mul(b.Mul(r, k), b.Weight(t + "bonus")))]);                  // [T, H, 1]
            y = b.Add(y, b.Reshape(b.Mul(bonus, Heads(v)), 0, hidden));
            if (b.HasWeight(t + "gate_down.weight"))
            {
                var gate = b.Linear(b.Sigmoid(b.Linear(xg, t + "gate_down.weight")), t + "gate_up.weight");
                y = b.Mul(y, gate);
            }

            return (b.Linear(y, t + "output.weight", name: $"{p}time_mix_out"), projectedValue);
        }

        Value ChannelMix(GraphBuilder b, Value h, string p, int il, Value selector)
        {
            string c = p + "channel_mix.";
            var previous = shift.Previous(b, h, selector, channelShift[il]);
            var mixed = b.Add(h, b.Mul(b.Sub(previous, h), b.Weight(c + "lerp.weight")));
            var activated = b.Unary("Square", b.Unary("Relu", b.Linear(mixed, c + "key.weight")));
            return b.Linear(activated, c + "value.weight", name: $"{p}channel_mix_out");
        }
    }
}
