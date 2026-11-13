using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// NVIDIA Nemotron-H family (canonical arch <c>nemotron_h</c>): a stack of residual blocks <c>x += mixer(norm(x))</c> whose mixer is, per layer,
/// a Mamba-2 state-space layer (<c>ssm</c>), position-less grouped-query attention (<c>attention</c>), a non-gated squared-ReLU MLP (<c>mlp</c>) or a
/// mixture of such MLPs (<c>moe</c>).
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>nemotron_h.*</c>): <c>layer_count</c>, <c>hidden_size</c>, <c>norm_eps</c> (all RMS norms incl. the gated SSM norm),
/// <c>context_length</c>, <c>block_kinds</c> (string[layer_count]: "ssm" | "attention" | "mlp" | "moe"), <c>mlp.activation</c> ("relu2" default, also "silu"/"gelu"/"relu"),
/// <c>attention.head_count</c>, <c>attention.kv_head_count</c>, <c>attention.head_dim</c>,
/// <c>ssm.conv_kernel</c>, <c>ssm.state_size</c> (N), <c>ssm.group_count</c> (G), <c>ssm.head_count</c> (H), <c>ssm.head_dim</c> (P; inner size = H·P),
/// optional <c>ssm.time_step_min</c> (lower clamp of softplus(dt); default 0); for "moe" layers <c>moe.expert_count</c>, <c>moe.expert_used_count</c>,
/// <c>moe.normalize_weights</c> (renormalize the selected scores; default true), <c>moe.weights_scale</c> (default 1).
/// Weights: <c>token_embedding.weight</c> [V, H], <c>output_norm.weight</c>, <c>lm_head.weight</c> (optional, tied otherwise); per layer <c>layers.{i}.input_norm.weight</c> and
/// <list type="bullet">
/// <item><c>ssm</c>: <c>ssm.in.weight</c> [inner + conv_dim + H, hidden] (rows: gate z | conv input x,B,C | dt), <c>ssm.conv1d.weight</c> [conv_dim, K] f32, <c>ssm.conv1d.bias</c> [conv_dim] f32,
/// <c>ssm.dt_bias</c> [H] f32, <c>ssm.a</c> [H] f32 (A = -exp(A_log), already negative), <c>ssm.d</c> [H] f32, <c>ssm.norm.weight</c> [inner] f32 (gated group RMS norm, G groups), <c>ssm.out.weight</c> [hidden, inner];
/// conv_dim = inner + 2·G·N.</item>
/// <item><c>attention</c>: <c>attention.{q,k,v,o}.weight</c> (no rope, no norms, scale 1/sqrt(head_dim)).</item>
/// <item><c>mlp</c>: <c>mlp.up.weight</c>, <c>mlp.down.weight</c>.</item>
/// <item><c>moe</c>: <c>moe.router.weight</c> [E, hidden], <c>moe.router_bias</c> [E] f32 (added to the sigmoid scores for the expert selection only),
/// rank-3 <c>moe.experts.{up,down}.weight</c> [E, I, hidden] / [E, hidden, I] (no gate), <c>moe.shared.{up,down}.weight</c> (an MLP every token passes through, added to the routed sum).
/// Weights of the k selected experts are their sigmoid scores (not the biased ones), renormalized to sum 1 and scaled.</item>
/// </list>
/// </remarks>
public sealed class NemotronHAdapter : IModelAdapter
{
    /// <inheritdoc/>
    public string Name => "Nemotron-H (Mamba-2 / attention / MLP hybrid)";
    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => ["nemotron_h"];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        int nLayer = src.Arch<int>("layer_count");
        int nEmbd = src.Arch<int>("hidden_size");
        float eps = src.Arch("norm_eps", 1e-5f);
        var kinds = src.Arch<string[]>("block_kinds");
        if (kinds.Length != nLayer)
        {
            throw new InvalidDataException("nemotron_h: block_kinds must have layer_count entries.");
        }

        string act = src.Arch("mlp.activation", "relu2");
        int nHead = src.Arch("attention.head_count", 0);
        int nKv = src.Arch("attention.kv_head_count", 0);
        int headDim = src.Arch("attention.head_dim", 0);
        int convK = src.Arch("ssm.conv_kernel", 4);
        int N = src.Arch("ssm.state_size", 0);
        int G = src.Arch("ssm.group_count", 1);
        int H = src.Arch("ssm.head_count", 0);
        int P = src.Arch("ssm.head_dim", 0);
        float dtMin = src.Arch("ssm.time_step_min", 0f);
        int experts = src.Arch("moe.expert_count", 0);
        int expertsUsed = src.Arch("moe.expert_used_count", 0);
        bool normWeights = src.Arch("moe.normalize_weights", true);
        float weightsScale = src.Arch("moe.weights_scale", 1f);
        int inner = H * P;
        int convDim = inner + 2 * G * N;
        int vocab = src.Weight("token_embedding.weight")!.Shape[0].IsSymbolic ? 0 : (int)src.Weight("token_embedding.weight")!.Shape[0].Fixed;

        var states = new List<StateDefinition>();
        var kCache = new Dictionary<int, StateDefinition>();
        var vCache = new Dictionary<int, StateDefinition>();
        var convState = new Dictionary<int, StateDefinition>();
        var ssmState = new Dictionary<int, StateDefinition>();
        for (int il = 0; il < nLayer; il++)
        {
            string g = $"layers.{il}";
            switch (kinds[il])
            {
                case "ssm":
                    states.Add(convState[il] = new StateDefinition { Name = $"{g}.conv_state", Type = DataType.F32, Shape = new Shape(convK - 1, convDim), Kind = StateKind.Recurrent, Group = g });
                    states.Add(ssmState[il] = new StateDefinition { Name = $"{g}.ssm_state", Type = DataType.F32, Shape = new Shape(H, P, N), Kind = StateKind.Recurrent, Group = g });
                    break;
                case "attention":
                    var shape = new Shape(Dim.Sym(WellKnown.ContextSymbol), nKv, headDim);
                    states.Add(kCache[il] = new StateDefinition { Name = $"{g}.k_cache", Type = options.KvCacheType, Shape = shape, Kind = StateKind.KeyCache, Group = g });
                    states.Add(vCache[il] = new StateDefinition { Name = $"{g}.v_cache", Type = options.KvCacheType, Shape = shape, Kind = StateKind.ValueCache, Group = g });
                    break;
                case "mlp" or "moe": break;
                default: throw new NotSupportedException($"nemotron_h: block kind '{kinds[il]}' is not supported.");
            }
        }

        var graphs = new Dictionary<string, ComputeGraph> { [WellKnown.DecodeGraph] = BuildTrunk(WellKnown.DecodeGraph, false) };
        if (options.IncludeAuxiliaryGraphs)
        {
            graphs["decode_embeddings"] = BuildTrunk("decode_embeddings", true);
        }

        var eos = new List<int>();
        if (src.Has(CanonicalKeys.TokenizerEosId))
        {
            eos.Add(src.Get<int>(CanonicalKeys.TokenizerEosId));
        }

        if (src.Has(CanonicalKeys.TokenizerEotId))
        {
            eos.Add(src.Get<int>(CanonicalKeys.TokenizerEotId));
        }

        return new ModelDefinition
        {
            Architecture = src.Architecture,
            Name = src.Get(CanonicalKeys.Name, src.Architecture),
            Weights = src.Weights,
            Graphs = graphs,
            States = states,
            SourceMetadata = src.Metadata,
            OwnedResources = src.OwnedResources,
            Properties = new Dictionary<string, object>
            {
                [WellKnown.VocabSize] = vocab,
                [WellKnown.ContextLength] = src.Arch("context_length", 32768),
                [WellKnown.EmbeddingLength] = nEmbd,
                [WellKnown.PositionAxes] = 1,
                [WellKnown.EosTokenIds] = eos.ToArray(),
                ["n_layer"] = nLayer,
            },
        };

        ComputeGraph BuildTrunk(string name, bool embeddingsInput)
        {
            var b = new GraphBuilder(name, src.Weight);
            var T = Dim.Sym(WellKnown.TokenSymbol);
            Value x;
            using (b.Group("input"))
            {
                if (embeddingsInput)
                {
                    x = b.Input("embeddings", InputRole.Embeddings, DataType.F32, new Shape(T, nEmbd));
                }
                else
                {
                    x = b.Gather(b.Weight("token_embedding.weight"), b.Input("tokens", InputRole.TokenIds, DataType.I32, new Shape(T)), "inp_embd");
                }
            }
            // Positions are an input of every graph (the runtime always feeds them) although no layer here uses rotary embeddings.
            b.Input("positions", InputRole.Positions, DataType.I32, new Shape(T, 1));
            var outRows = b.Input("output_rows", InputRole.OutputRows, DataType.I32, new Shape(Dim.Sym("R")));

            for (int il = 0; il < nLayer; il++)
            {
                using var _ = b.Group($"layers.{il}");
                string p = $"layers.{il}.";
                var h = b.RmsNorm(x, b.Weight(p + "input_norm.weight"), eps, $"{p}norm");
                var y = kinds[il] switch
                {
                    "ssm" => Ssm(b, h, p, il),
                    "attention" => Attention(b, h, p, il),
                    "moe" => Moe(b, h, p),
                    _ => Mlp(b, h, p),
                };
                x = b.Add(y, x, $"{p}l_out");
            }

            using (b.Group("output"))
            {
                var hidden = b.RmsNorm(x, b.Weight("output_norm.weight"), eps, "result_norm");
                var rows = b.Gather(hidden, outRows, "selected_rows");
                var head = b.TryWeight("lm_head.weight") ?? b.Weight("token_embedding.weight");
                b.Output("logits", b.MatMul(rows, head, "logits"));
                b.Output("hidden", hidden);
            }
            return b.Build();
        }

        Value Ssm(GraphBuilder b, Value h, string p, int il)
        {
            var proj = b.Linear(h, p + "ssm.in.weight");                                   // [T, inner + conv_dim + H]
            var z = b.Slice(proj, 1, 0, inner);
            var xbc = b.Slice(proj, 1, inner, convDim);
            var dt = b.Slice(proj, 1, inner + convDim, H);
            var cs = b.State(convState[il]);
            var ss = b.State(ssmState[il]);
            var conv = b.Emit("CausalConv1d", [xbc, b.Weight(p + "ssm.conv1d.weight"), cs, b.Weight(p + "ssm.conv1d.bias")],
                Attributes.From(("activation", "silu"), ("bias", true)), writes: [cs])[0];
            var xs = b.Reshape(b.Slice(conv, 1, 0, inner), 0, H, P);
            var bs = b.Reshape(b.Slice(conv, 1, inner, G * N), 0, G, N);
            var cc = b.Reshape(b.Slice(conv, 1, inner + G * N, G * N), 0, G, N);
            var scan = b.Emit("SelectiveScan", [xs, dt, b.Weight(p + "ssm.dt_bias"), b.Weight(p + "ssm.a"), bs, cc, b.Weight(p + "ssm.d"), ss],
                Attributes.From(("dt_min", dtMin)), $"{p}scan", writes: [ss])[0];                // [T, H, P]
            // gated group RMS norm: norm_G(y · silu(z)) · weight
            var gated = b.Mul(b.Reshape(scan, 0, inner), b.Silu(z));
            var normed = b.Reshape(b.RmsNorm(b.Reshape(gated, 0, G, inner / G), null, eps), 0, inner);
            normed = b.Mul(normed, b.Weight(p + "ssm.norm.weight"), $"{p}gated_norm");
            return b.Linear(normed, p + "ssm.out.weight", name: $"{p}ssm_out");
        }

        Value Attention(GraphBuilder b, Value h, string p, int il)
        {
            var q = b.Reshape(b.Linear(h, p + "attention.q.weight"), 0, nHead, headDim);
            var k = b.Reshape(b.Linear(h, p + "attention.k.weight"), 0, nKv, headDim);
            var v = b.Reshape(b.Linear(h, p + "attention.v.weight"), 0, nKv, headDim);
            b.CacheWrite(b.State(kCache[il]), k);
            b.CacheWrite(b.State(vCache[il]), v);
            var attn = b.CachedAttention(q, b.State(kCache[il]), b.State(vCache[il]), 1f / MathF.Sqrt(headDim), name: $"{p}attn");
            return b.Linear(b.Reshape(attn, 0, nHead * headDim), p + "attention.o.weight", name: $"{p}attn_out");
        }

        Value Act(GraphBuilder b, Value up) => act switch
        {
            "relu2" => b.Unary("Square", b.Unary("Relu", up)),
            "silu" => b.Silu(up),
            "gelu" => b.Unary("Gelu", up),
            "relu" => b.Unary("Relu", up),
            _ => throw new NotSupportedException($"nemotron_h: unknown mlp.activation '{act}'."),
        };

        Value Mlp(GraphBuilder b, Value h, string p, string prefix = "mlp.")
            => b.Linear(Act(b, b.Linear(h, p + prefix + "up.weight")), p + prefix + "down.weight", name: $"{p}ffn_out");

        // Sigmoid-scored routing with a selection-only bias, k experts per token (non-gated MLPs), plus the shared MLP.
        Value Moe(GraphBuilder b, Value h, string p)
        {
            if (experts <= 0 || expertsUsed <= 0)
            {
                throw new InvalidDataException("nemotron_h: moe layers need moe.expert_count and moe.expert_used_count.");
            }

            var scores = b.Sigmoid(b.Linear(h, p + "moe.router.weight"), $"{p}router_scores");                 // [T, E]
            var choice = b.TryWeight(p + "moe.router_bias") is { } bias ? b.Add(scores, bias) : scores;
            var top = b.Emit("TopK", [choice], Attributes.From(("k", (long)expertsUsed)), $"{p}router_topk");
            var ids = top[1];                                                                                     // [T, k] i32
            // The weights are the unbiased scores of the selected experts: one-hot pick (ids == iota) instead of a gather along the expert axis.
            var iota = b.Constant($"{p}expert_iota", DataType.F32, new Shape(experts), IotaBytes(experts));
            var onehot = b.Binary("Equal", b.Reshape(b.Emit1("Cast", [ids], Attributes.From(("type", "f32"))), 0, expertsUsed, 1), iota);   // [T, k, E]
            var picked = b.Reshape(b.Emit1("ReduceSum", [b.Mul(onehot, b.Reshape(scores, 0, 1, experts))]), 0, expertsUsed);               // [T, k]
            var weights = picked;
            if (normWeights)
            {
                weights = b.Binary("Div", weights, b.Scale(b.Emit1("ReduceSum", [weights]), 1f, 1e-20f));
            }

            if (weightsScale != 1f)
            {
                weights = b.Scale(weights, weightsScale);
            }

            var up = b.Emit1("MatMulId", [h, b.Weight(p + "moe.experts.up.weight"), ids], null, $"{p}experts_up");                         // [T, k, I]
            var down = b.Emit1("MatMulId", [Act(b, up), b.Weight(p + "moe.experts.down.weight"), ids], null, $"{p}experts_down");         // [T, k, H]
            var summed = b.Emit1("ReduceSum", [b.Permute(b.Mul(down, b.Reshape(weights, 0, expertsUsed, 1)), 0, 2, 1)]);                    // [T, H, 1]
            var routed = b.Reshape(summed, 0, nEmbd);
            return b.HasWeight(p + "moe.shared.up.weight") ? b.Add(routed, Mlp(b, h, p, "moe.shared.")) : routed;
        }

        static byte[] IotaBytes(int n)
        {
            var bytes = new byte[n * 4];
            for (int i = 0; i < n; i++)
            {
                BitConverter.TryWriteBytes(bytes.AsSpan(i * 4, 4), (float)i);
            }

            return bytes;
        }
    }
}
