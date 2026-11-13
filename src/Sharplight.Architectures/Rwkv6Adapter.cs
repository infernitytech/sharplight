using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// RWKV-6 "Finch" (canonical arch <c>rwkv6</c>): attention-free recurrent language model. Every layer is a time-mix block (token-shift with data-dependent interpolation
/// and decay from low-rank projections, the <c>Wkv6</c> recurrence, per-head group norm, SiLU output gate) followed by a channel-mix block (token-shift, squared-ReLU MLP
/// gated by a sigmoid receptance). All norms are LayerNorms with bias.
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>rwkv6.*</c>): <c>layer_count</c>, <c>hidden_size</c>, <c>head_size</c> (N; heads = hidden / N), <c>norm_eps</c>, <c>context_length</c>,
/// <c>rescale_every_layers</c> (the residual stream is halved after every n-th layer, a half-precision overflow guard; 0 = never), <c>time_mix_rank</c> (D, per interpolation).
/// Weights: <c>token_embedding.weight</c> [V, C], <c>embedding_norm.{weight,bias}</c>, <c>output_norm.{weight,bias}</c>, <c>lm_head.weight</c>; per layer <c>input_norm.{weight,bias}</c>,
/// <c>post_attention_norm.{weight,bias}</c>, and below <c>time_mix.</c>: <c>lerp_x.weight</c> [C], <c>lerp_{w,k,v,r,g}.weight</c> [C], <c>mix_down.weight</c> [5·D, C],
/// <c>mix_up.{0..4}.weight</c> [C, D] (interpolation offsets for w, k, v, r, g), <c>decay_bias</c> [C], <c>decay_down.weight</c> [Dd, C], <c>decay_up.weight</c> [C, Dd],
/// <c>bonus</c> [H, N], <c>{receptance,key,value,gate,output}.weight</c> [C, C], <c>out_norm.{weight,bias}</c> [C] (group norm affine);
/// below <c>channel_mix.</c>: <c>lerp_k.weight</c>, <c>lerp_r.weight</c> [C], <c>key.weight</c> [I, C], <c>value.weight</c> [C, I], <c>receptance.weight</c> [C, C].
/// Per-layer states: the last time-mix and channel-mix inputs (token shift) and the [H, N, N] wkv state.
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
            for (int il = 0; il < layerCount; il++)
            {
                using var _ = b.Group($"layers.{il}");
                string p = $"layers.{il}.";
                x = b.Add(TimeMix(b, LayerNorm(b, x, p + "input_norm"), p, il, selector), x, $"{p}attn_out");
                x = b.Add(ChannelMix(b, LayerNorm(b, x, p + "post_attention_norm"), p, il, selector), x, $"{p}l_out");
                if (rescaleEvery > 0 && (il + 1) % rescaleEvery == 0)
                {
                    x = b.Scale(x, 0.5f);
                }
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
            var r = b.Linear(xr, t + "receptance.weight");
            var k = b.Linear(xk, t + "key.weight");
            var v = b.Linear(xv, t + "value.weight");
            var gate = b.Silu(b.Linear(xg, t + "gate.weight"));
            var decayLog = b.Add(b.Weight(t + "decay_bias"), b.Linear(b.Unary("Tanh", b.Linear(xw, t + "decay_down.weight")), t + "decay_up.weight"));
            var w = b.Unary("Exp", b.Unary("Neg", b.Unary("Exp", decayLog)));

            Value Heads(Value z) => b.Reshape(z, 0, heads, headSize);
            var ws = b.State(wkvState[il]);
            var wkv = b.Emit("Wkv6", [Heads(r), Heads(k), Heads(v), Heads(w), b.Reshape(b.Weight(t + "bonus"), heads, headSize), ws], null, $"{p}wkv", writes: [ws])[0];
            var normed = b.Emit1("LayerNorm", [wkv], Attributes.From(("eps", GroupNormEps)));         // [T, H, N], statistics per head
            var y = b.Add(b.Mul(b.Reshape(normed, 0, hidden), b.Weight(t + "out_norm.weight")), b.Weight(t + "out_norm.bias"));
            return b.Linear(b.Mul(y, gate), t + "output.weight", name: $"{p}time_mix_out");
        }

        Value ChannelMix(GraphBuilder b, Value h, string p, int il, Value selector)
        {
            string c = p + "channel_mix.";
            var delta = b.Sub(shift.Previous(b, h, selector, channelShift[il]), h);
            var xk = b.Add(h, b.Mul(delta, b.Weight(c + "lerp_k.weight")));
            var xr = b.Add(h, b.Mul(delta, b.Weight(c + "lerp_r.weight")));
            var activated = b.Unary("Square", b.Unary("Relu", b.Linear(xk, c + "key.weight")));
            var receptance = b.Sigmoid(b.Linear(xr, c + "receptance.weight"));
            return b.Mul(receptance, b.Linear(activated, c + "value.weight", name: $"{p}channel_mix_out"));
        }
    }
}
