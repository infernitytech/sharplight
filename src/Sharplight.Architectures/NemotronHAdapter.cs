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
        var mamba = new Mamba2Layer(src, eps);
        // Sigmoid-scored routing (with a selection-only bias), whatever moe.router_activation says.
        var moe = MoeOptions.Read(src, normalizeDefault: true) with { ScoreFunction = MoeScoreFunction.Sigmoid };
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
                    (convState[il], ssmState[il]) = mamba.DefineStates(g);
                    states.Add(convState[il]);
                    states.Add(ssmState[il]);
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
                    "ssm" => mamba.Build(b, h, p, convState[il], ssmState[il]),
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

        // k experts per token (non-gated MLPs) plus the shared MLP.
        Value Moe(GraphBuilder b, Value h, string p) => MoeBlock.Build(b, h, p + "moe.", moe, nEmbd, t => Act(b, t));
    }
}
