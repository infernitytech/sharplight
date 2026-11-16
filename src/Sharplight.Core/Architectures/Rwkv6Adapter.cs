using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// RWKV-6 "Finch" (canonical arch <c>rwkv6</c>): attention-free recurrent language model. Every layer is a time-mix block (token-shift with data-dependent interpolation
/// and decay from low-rank projections, the <c>Wkv6</c> recurrence, per-head group norm, SiLU output gate) followed by a channel-mix block (token-shift, squared-ReLU MLP
/// gated by a sigmoid receptance). All norms are LayerNorms with bias. The QRWKV6 distillations of Transformer models (llama.cpp <c>rwkv6qwen2</c>) share the
/// interpolation and decay computation but use RMS norms, no embedding norm, biased receptance / key / value projections with grouped key/value heads, a sigmoid
/// output gate, a gated-linear-attention recurrence without bonus or group norm, and a SwiGLU MLP in place of the channel-mix.
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>rwkv6.*</c>): <c>layer_count</c>, <c>hidden_size</c>, <c>head_size</c> (N; heads = hidden / N), <c>norm_eps</c>, <c>context_length</c>,
/// <c>rescale_every_layers</c> (the residual stream is halved after every n-th layer, a half-precision overflow guard; 0 = never), <c>time_mix_rank</c> (D, per interpolation),
/// <c>norm_type</c> ("layer" (default, weight and bias) or "rms" (weight only)).
/// Weights: <c>token_embedding.weight</c> [V, C], <c>embedding_norm.{weight,bias}</c>, <c>output_norm.{weight,bias}</c>, <c>lm_head.weight</c>; per layer <c>input_norm.{weight,bias}</c>,
/// <c>post_attention_norm.{weight,bias}</c>, and below <c>time_mix.</c>: <c>lerp_x.weight</c> [C], <c>lerp_{w,k,v,r,g}.weight</c> [C], <c>mix_down.weight</c> [5·D, C],
/// <c>mix_up.{0..4}.weight</c> [C, D] (interpolation offsets for w, k, v, r, g), <c>decay_bias</c> [C], <c>decay_down.weight</c> [Dd, C], <c>decay_up.weight</c> [C, Dd],
/// <c>bonus</c> [H, N], <c>{receptance,key,value,gate,output}.weight</c> [C, C], <c>out_norm.{weight,bias}</c> [C] (group norm affine);
/// below <c>channel_mix.</c>: <c>lerp_k.weight</c>, <c>lerp_r.weight</c> [C], <c>key.weight</c> [I, C], <c>value.weight</c> [C, I], <c>receptance.weight</c> [C, C].
/// QRWKV6 layers have no <c>bonus</c> and no <c>out_norm</c> (that selects the gated-linear-attention form), optional <c>{receptance,key,value}.bias</c>,
/// <c>key.weight</c> / <c>value.weight</c> with [Hkv·N, C] rows (each key/value head serves H / Hkv consecutive heads), and <c>mlp.{gate,up,down}.weight</c> instead of the channel-mix.
/// Per-layer states: the last time-mix input and, with the channel-mix, its last input (token shift) and the [H, N, N] wkv state.
/// </remarks>
public sealed class Rwkv6Adapter : IModelAdapter
{
    private const float GroupNormEps = 64e-5f;
    private static readonly string[] s_mixes = ["w", "k", "v", "r", "g"];

    /// <inheritdoc/>
    public string Name => "RWKV-6";
    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => ["rwkv6"];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        int layerCount = src.Arch<int>("layer_count");
        int hidden = src.Arch<int>("hidden_size");
        int headSize = src.Arch("head_size", 64);
        int heads = hidden / headSize;
        int rank = src.Arch("time_mix_rank", 32);
        int rescaleEvery = src.Arch("rescale_every_layers", 0);
        float eps = src.Arch("norm_eps", 1e-5f);
        bool rmsNorm = src.Arch("norm_type", "layer") == "rms";
        bool swiglu = src.Weight("layers.0.mlp.gate.weight") is not null;
        var shift = new RwkvTokenShift(hidden);

        var states = new List<StateDefinition>();
        var timeShift = new StateDefinition[layerCount];
        var channelShift = new StateDefinition?[layerCount];
        var wkvState = new StateDefinition[layerCount];
        for (int il = 0; il < layerCount; il++)
        {
            string g = $"layers.{il}";
            timeShift[il] = shift.DefineState($"{g}.time_shift", g);
            channelShift[il] = swiglu ? null : shift.DefineState($"{g}.channel_shift", g);
            wkvState[il] = new StateDefinition { Name = $"{g}.wkv_state", Type = DataType.F32, Shape = new Shape(heads, headSize, headSize), Kind = StateKind.Recurrent, Group = g };
            states.AddRange(swiglu ? [timeShift[il], wkvState[il]] : [timeShift[il], channelShift[il]!, wkvState[il]]);
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
                x = b.HasWeight("embedding_norm.weight") ? Norm(b, x, "embedding_norm") : x;
            }

            b.Input("positions", InputRole.Positions, DataType.I32, new Shape(T, 1));
            var outRows = b.Input("output_rows", InputRole.OutputRows, DataType.I32, new Shape(Dim.Sym("R")));
            var selector = shift.Selector(b);
            for (int il = 0; il < layerCount; il++)
            {
                using var _ = b.Group($"layers.{il}");
                string p = $"layers.{il}.";
                x = b.Add(TimeMix(b, Norm(b, x, p + "input_norm"), p, il, selector), x, $"{p}attn_out");
                var mlpInput = Norm(b, x, p + "post_attention_norm");
                x = b.Add(swiglu ? SwiGlu(b, mlpInput, p) : ChannelMix(b, mlpInput, p, il, selector), x, $"{p}l_out");
                if (rescaleEvery > 0 && (il + 1) % rescaleEvery == 0)
                {
                    x = b.Scale(x, 0.5f);
                }
            }

            using (b.Group("output"))
            {
                var hiddenStates = Norm(b, x, "output_norm");
                var rows = b.Gather(hiddenStates, outRows, "selected_rows");
                var head = b.TryWeight("lm_head.weight") ?? b.Weight("token_embedding.weight");
                b.Output("logits", b.MatMul(rows, head, "logits"));
                b.Output("hidden", hiddenStates);
            }

            return b.Build();
        }

        Value Norm(GraphBuilder b, Value v, string prefix)
            => rmsNorm
                ? b.RmsNorm(v, b.Weight(prefix + ".weight"), eps)
                : b.Emit1("LayerNorm", [v, b.Weight(prefix + ".weight"), b.Weight(prefix + ".bias")], Attributes.From(("eps", eps)));

        Value SwiGlu(GraphBuilder b, Value h, string p)
            => b.Linear(b.Mul(b.Silu(b.Linear(h, p + "mlp.gate.weight")), b.Linear(h, p + "mlp.up.weight")), p + "mlp.down.weight", name: $"{p}mlp_out");

        Value TimeMix(GraphBuilder b, Value h, string p, int il, Value selector)
        {
            string t = p + "time_mix.";
            var delta = b.Sub(shift.Previous(b, h, selector, timeShift[il]), h);
            var baseMix = b.Add(h, b.Mul(delta, b.Weight(t + "lerp_x.weight")));
            var lowRank = b.Unary("Tanh", b.Linear(baseMix, t + "mix_down.weight"));                  // [T, 5·D]
            Value Mixed(int index)
            {
                var offset = b.Linear(b.Slice(lowRank, 1, (long)index * rank, rank), $"{t}mix_up.{index}.weight");
                return b.Add(h, b.Mul(delta, b.Add(b.Weight($"{t}lerp_{s_mixes[index]}.weight"), offset)));
            }

            var xw = Mixed(0);
            var xk = Mixed(1);
            var xv = Mixed(2);
            var xr = Mixed(3);
            var xg = Mixed(4);
            var r = b.Linear(xr, t + "receptance.weight", BiasOf(b, t + "receptance.bias"));
            var k = b.Linear(xk, t + "key.weight", BiasOf(b, t + "key.bias"));
            var v = b.Linear(xv, t + "value.weight", BiasOf(b, t + "value.bias"));
            var decayLog = b.Add(b.Weight(t + "decay_bias"), b.Linear(b.Unary("Tanh", b.Linear(xw, t + "decay_down.weight")), t + "decay_up.weight"));
            var w = b.Unary("Exp", b.Unary("Neg", b.Unary("Exp", decayLog)));

            Value Heads(Value z) => b.Reshape(z, 0, heads, headSize);
            var ws = b.State(wkvState[il]);
            if (!b.HasWeight(t + "bonus"))
            {
                var noBonus = b.Scale(b.Reshape(b.Weight(t + "decay_bias"), heads, headSize), 0f);
                var attended = GatedLinearAttention(b, Heads(r), ExpandKeyValueHeads(b, k), ExpandKeyValueHeads(b, v), Heads(w), noBonus, ws, $"{p}wkv");
                var output = b.Mul(b.Reshape(attended, 0, hidden), b.Sigmoid(b.Linear(xg, t + "gate.weight")));
                return b.Linear(output, t + "output.weight", name: $"{p}time_mix_out");
            }

            var gate = b.Silu(b.Linear(xg, t + "gate.weight"));
            var wkv = b.Emit("Wkv6", [Heads(r), Heads(k), Heads(v), Heads(w), b.Reshape(b.Weight(t + "bonus"), heads, headSize), ws], null, $"{p}wkv", writes: [ws])[0];
            var normed = b.Emit1("LayerNorm", [wkv], Attributes.From(("eps", GroupNormEps)));         // [T, H, N], statistics per head
            var y = b.Add(b.Mul(b.Reshape(normed, 0, hidden), b.Weight(t + "out_norm.weight")), b.Weight(t + "out_norm.bias"));
            return b.Linear(b.Mul(y, gate), t + "output.weight", name: $"{p}time_mix_out");
        }

        string? BiasOf(GraphBuilder b, string name) => b.HasWeight(name) ? name : null;

        // Grouped key/value heads: every one of the Hkv heads serves H / Hkv consecutive heads. [T, Hkv·N] -> [T, H, N].
        Value ExpandKeyValueHeads(GraphBuilder b, Value z)
        {
            int kvHeads = (int)z.Shape[1].Fixed / headSize;
            if (kvHeads == heads)
            {
                return b.Reshape(z, 0, heads, headSize);
            }

            var perHead = b.Reshape(z, 0, kvHeads, 1, headSize);
            var copies = Enumerable.Repeat(perHead, heads / kvHeads).ToArray();
            return b.Reshape(b.Concat(2, copies), 0, heads, headSize);
        }

        // Gated linear attention, S = diag(w)·S + k ⊗ v and y = scale · r · S (the state after the current token), on the Wkv6 recurrence: with the bonus
        // zeroed and the receptance premultiplied by the decay, Wkv6 yields scale · (r ⊙ w) · S(previous); the current token's own k ⊗ v term is added to it
        // directly. The key is damped by the decay (k · (1 - w)) as in QRWKV6. All operands are [T, H, N].
        Value GatedLinearAttention(GraphBuilder b, Value r, Value k, Value v, Value w, Value noBonus, Value state, string name)
        {
            float scale = 1f / MathF.Sqrt(headSize);
            var damped = b.Sub(k, b.Mul(k, w));
            var previous = b.Emit("Wkv6", [b.Scale(b.Mul(r, w), scale), damped, v, w, noBonus, state], null, name, writes: [state])[0];
            var current = b.Mul(b.Emit1("ReduceSum", [b.Scale(b.Mul(r, damped), scale)]), v);           // (Σ r·k) · v per head
            return b.Add(previous, current);
        }

        Value ChannelMix(GraphBuilder b, Value h, string p, int il, Value selector)
        {
            string c = p + "channel_mix.";
            var delta = b.Sub(shift.Previous(b, h, selector, channelShift[il]!), h);
            var xk = b.Add(h, b.Mul(delta, b.Weight(c + "lerp_k.weight")));
            var xr = b.Add(h, b.Mul(delta, b.Weight(c + "lerp_r.weight")));
            var activated = b.Unary("Square", b.Unary("Relu", b.Linear(xk, c + "key.weight")));
            var receptance = b.Sigmoid(b.Linear(xr, c + "receptance.weight"));
            return b.Mul(receptance, b.Linear(activated, c + "value.weight", name: $"{p}channel_mix_out"));
        }
    }
}
