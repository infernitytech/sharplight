using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;
using Sharplight.Vision;

namespace Sharplight.Architectures;

/// <summary>
/// Qwen3.5 / Qwen3.6 / Qwen3.8 dense hybrid architecture (canonical arch <c>qwen3_5</c>):
/// every <c>full_attention_interval</c>-th layer is gated softmax attention with partial interleaved
/// M-RoPE; all others are Gated DeltaNet linear-attention layers with a short causal conv.
/// Optional MTP layers are exposed as an auxiliary draft graph later.
/// </summary>
/// <remarks>
/// Canonical description consumed (importers translate their formats into exactly this):
/// <list type="bullet">
/// <item>Hyper-parameters (<c>qwen3_5.*</c>): <c>layer_count</c> (trunk, without MTP), <c>hidden_size</c>,
/// <c>ffn_hidden_size</c>, <c>context_length</c>, <c>norm_eps</c>, <c>attention.head_count</c>,
/// <c>attention.kv_head_count</c>, <c>attention.head_dim</c>, <c>rope.dim</c>, <c>rope.base</c>,
/// <c>rope.sections</c> (4 M-RoPE axes), <c>full_attention_interval</c>, <c>linear_attention.conv_kernel</c>,
/// <c>linear_attention.key_head_count</c>, <c>linear_attention.key_head_dim</c>,
/// <c>linear_attention.value_head_count</c>, <c>linear_attention.value_head_dim</c>, <c>mtp.layer_count</c>.</item>
/// <item>Weights: <c>token_embedding.weight</c>, <c>output_norm.weight</c>, <c>lm_head.weight</c> (optional, tied
/// to the embedding otherwise); per layer <c>layers.{i}.</c>: <c>input_norm</c>, <c>post_attention_norm</c>,
/// <c>mlp.{gate,up,down}</c>; attention layers <c>attention.{q,k,v,o,q_norm,k_norm}</c> (q rows interleave
/// [query | output gate] per head); linear-attention layers <c>linear_attention.{qkv,z,beta,alpha,out,norm,conv1d}</c>
/// (all <c>.weight</c>), <c>linear_attention.a</c> (A = -exp(A_log)) and <c>linear_attention.dt_bias</c>.
/// MTP: <c>mtp.layers.{j}.*</c> (same per-layer names), <c>mtp.{fc,embedding_norm,hidden_norm,output_norm}.weight</c>.</item>
/// <item>Semantics: RMS norm weights multiply directly (any "1 + w" convention already folded in); value heads in
/// tiled order; conv1d weights [channels, kernel] f32.</item>
/// <item>Optional vision tower (image input): hyper-parameters <c>qwen3_5.vision.*</c> and weights <c>vision.*</c> exactly as
/// documented on <see cref="VisionParameters"/>, found either in the primary source (e.g. Hugging Face checkpoints) or in an
/// auxiliary source with role <see cref="AuxiliaryRoles.Mmproj"/> (e.g. a separate projector file); the adapter does not care
/// which file they came from. With a tower the definition also has the <see cref="WellKnown.VisionEncodeGraph"/> and
/// <see cref="WellKnown.DecodeMultimodalGraph"/> graphs and <c>vision.*</c> properties; prompt assembly is
/// <see cref="Qwen35Multimodal"/>.</item>
/// </list>
/// </remarks>
public sealed class Qwen35Adapter : IModelAdapter
{
    /// <inheritdoc/>
    public string Name => "Qwen3.5 hybrid (Gated DeltaNet + gated attention)";
    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => ["qwen3_5"];

    private sealed record HParams(
        int NLayer, int NEmbd, int NFf, int NHead, int NHeadKv, int HeadDim, float Eps,
        int RopeDims, float RopeBase, int[] RopeSections,
        int ConvKernel, int SsmState, int SsmGroups, int SsmDtRank, int SsmInner, int FullAttnInterval,
        int Vocab, int ContextLength, bool MRope, bool GroupedValueHeads, MoeOptions Moe);

    private const int MRopeAxes = 4;

    // [T, heads, dim] → [T, heads·repeats, dim] with each head repeated <repeats> times consecutively.
    private static Value RepeatHeadsInPlace(GraphBuilder b, Value x, int heads, int repeats, int dim)
    {
        var tiled = b.Emit1("Tile", [b.Reshape(x, 0, heads, 1, dim)], Attributes.From(("repeats", new long[] { 1, 1, repeats, 1 })));
        return b.Reshape(tiled, 0, (long)heads * repeats, dim);
    }

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        int nextn = src.Arch("mtp.layer_count", 0);
        int valueHeads = src.Arch<int>("linear_attention.value_head_count");
        var hp = new HParams(
            NLayer: src.Arch<int>("layer_count"),
            NEmbd: src.Arch<int>("hidden_size"),
            NFf: src.Arch<int>("ffn_hidden_size"),
            NHead: src.Arch<int>("attention.head_count"),
            NHeadKv: src.Arch<int>("attention.kv_head_count"),
            HeadDim: src.Arch<int>("attention.head_dim"),
            Eps: src.Arch<float>("norm_eps"),
            RopeDims: src.Arch<int>("rope.dim"),
            RopeBase: src.Arch("rope.base", 10000f),
            RopeSections: src.Arch("rope.sections", Array.Empty<int>()),
            ConvKernel: src.Arch<int>("linear_attention.conv_kernel"),
            SsmState: src.Arch<int>("linear_attention.key_head_dim"),
            SsmGroups: src.Arch<int>("linear_attention.key_head_count"),
            SsmDtRank: valueHeads,
            SsmInner: src.Arch<int>("linear_attention.value_head_dim") * valueHeads,
            FullAttnInterval: src.Arch("full_attention_interval", 4),
            Vocab: src.Weight("token_embedding.weight")!.Shape[0].IsSymbolic ? 0 : (int)src.Weight("token_embedding.weight")!.Shape[0].Fixed,
            ContextLength: src.Arch("context_length", 32768),
            MRope: src.Has($"{src.Architecture}.rope.sections"),
            GroupedValueHeads: src.Arch("linear_attention.grouped_value_heads", false),
            Moe: MoeOptions.Read(src, normalizeDefault: true));

        bool IsRecurrent(int il) => (il + 1) % hp.FullAttnInterval != 0;
        int positionAxes = hp.MRope ? MRopeAxes : 1;

        // ---- persistent state ---------------------------------------------------------------
        var states = new List<StateDefinition>();
        var kCache = new Dictionary<int, StateDefinition>();
        var vCache = new Dictionary<int, StateDefinition>();
        var convState = new Dictionary<int, StateDefinition>();
        var ssmState = new Dictionary<int, StateDefinition>();
        int convChannels = hp.SsmInner + 2 * hp.SsmGroups * hp.SsmState;
        int headVDim = hp.SsmInner / hp.SsmDtRank;
        for (int il = 0; il < hp.NLayer; il++)
        {
            string g = $"layers.{il}";
            if (IsRecurrent(il))
            {
                states.Add(convState[il] = new StateDefinition { Name = $"{g}.conv_state", Type = DataType.F32, Shape = new Shape(hp.ConvKernel - 1, convChannels), Kind = StateKind.Recurrent, Group = g });
                states.Add(ssmState[il] = new StateDefinition { Name = $"{g}.ssm_state", Type = DataType.F32, Shape = new Shape(hp.SsmDtRank, headVDim, hp.SsmState), Kind = StateKind.Recurrent, Group = g });
            }
            else
            {
                states.Add(kCache[il] = new StateDefinition { Name = $"{g}.k_cache", Type = options.KvCacheType, Shape = new Shape(Dim.Sym(WellKnown.ContextSymbol), hp.NHeadKv, hp.HeadDim), Kind = StateKind.KeyCache, Group = g });
                states.Add(vCache[il] = new StateDefinition { Name = $"{g}.v_cache", Type = options.KvCacheType, Shape = new Shape(Dim.Sym(WellKnown.ContextSymbol), hp.NHeadKv, hp.HeadDim), Kind = StateKind.ValueCache, Group = g });
            }
        }

        // Vision tower: in the primary source or an auxiliary "mmproj" source, same canonical names either way.
        var visionSrc = sources.Auxiliary.GetValueOrDefault(AuxiliaryRoles.Mmproj) ?? src;
        var vision = VisionParameters.FromMetadata(visionSrc.Metadata, visionSrc.Architecture + ".vision.");
        if (vision is not null && !string.Equals(visionSrc.Architecture, src.Architecture, StringComparison.OrdinalIgnoreCase))
        {
            throw new NotSupportedException($"The vision tower source '{visionSrc.Path}' belongs to architecture '{visionSrc.Architecture}', not '{src.Architecture}'.");
        }

        var allWeights = src.Weights;
        if (vision is not null && !ReferenceEquals(visionSrc, src))
        {
            var merged = new Dictionary<string, WeightInfo>(src.Weights, StringComparer.Ordinal);
            foreach (var (k, w) in visionSrc.Weights)
            {
                merged[k] = w;
            }

            allWeights = merged;
        }
        WeightInfo? Lookup(string n) => allWeights.TryGetValue(n, out var w) ? w : null;
        var graphs = new Dictionary<string, ComputeGraph>
        {
            [WellKnown.DecodeGraph] = BuildTrunk(WellKnown.DecodeGraph, embeddingsInput: false),
        };
        if (options.IncludeAuxiliaryGraphs)
        {
            graphs["decode_embeddings"] = BuildTrunk("decode_embeddings", embeddingsInput: true);
        }

        if (vision is not null)
        {
            graphs[WellKnown.DecodeMultimodalGraph] = BuildTrunk(WellKnown.DecodeMultimodalGraph, embeddingsInput: true, mixed: true);
            graphs[WellKnown.VisionEncodeGraph] = VisionTowerGraph.Build(vision, Lookup);
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

        var props = new Dictionary<string, object>
        {
            [WellKnown.VocabSize] = hp.Vocab,
            [WellKnown.ContextLength] = hp.ContextLength,
            [WellKnown.EmbeddingLength] = hp.NEmbd,
            [WellKnown.PositionAxes] = positionAxes,
            [WellKnown.EosTokenIds] = eos.ToArray(),
            ["n_layer"] = hp.NLayer,
            ["mtp_layers"] = nextn,
        };
        if (vision is not null)
        {
            if (vision.OutHiddenSize != hp.NEmbd)
            {
                throw new InvalidDataException($"The vision tower produces {vision.OutHiddenSize}-wide embeddings but the language model's hidden size is {hp.NEmbd}.");
            }

            vision.WriteProperties(props);
        }
        var owned = new List<IDisposable>(src.OwnedResources);
        if (!ReferenceEquals(visionSrc, src))
        {
            owned.AddRange(visionSrc.OwnedResources);
        }

        return new ModelDefinition
        {
            Architecture = src.Architecture,
            Name = src.Get(CanonicalKeys.Name, src.Architecture),
            Weights = allWeights,
            Graphs = graphs,
            States = states,
            SourceMetadata = src.Metadata,
            OwnedResources = owned,
            Properties = props,
        };

        ComputeGraph BuildTrunk(string name, bool embeddingsInput, bool mixed = false)
        {
            var b = new GraphBuilder(name, Lookup);
            var T = Dim.Sym(WellKnown.TokenSymbol);
            Value x;
            using (b.Group("input"))
            {
                if (mixed)
                {
                    // Rows are token embeddings or media embeddings: x = tok·(1-mask) + emb·mask.
                    var tokens = b.Input("tokens", InputRole.TokenIds, DataType.I32, new Shape(T));
                    var emb = b.Input("embeddings", InputRole.Embeddings, DataType.F32, new Shape(T, hp.NEmbd));
                    var mask = b.Input("embedding_mask", InputRole.EmbeddingMask, DataType.F32, new Shape(T, 1));
                    var tok = b.Gather(b.Weight("token_embedding.weight"), tokens, "tok_embd");
                    x = b.Add(b.Mul(tok, b.Scale(mask, -1f, 1f)), b.Mul(emb, mask), "inp_embd");
                }
                else if (embeddingsInput)
                {
                    x = b.Input("embeddings", InputRole.Embeddings, DataType.F32, new Shape(T, hp.NEmbd));
                }
                else
                {
                    var tokens = b.Input("tokens", InputRole.TokenIds, DataType.I32, new Shape(T));
                    x = b.Gather(b.Weight("token_embedding.weight"), tokens, "inp_embd");
                }
            }
            var positions = b.Input("positions", InputRole.Positions, DataType.I32, new Shape(T, positionAxes));
            var outRows = b.Input("output_rows", InputRole.OutputRows, DataType.I32, new Shape(Dim.Sym("R")));

            for (int il = 0; il < hp.NLayer; il++)
            {
                using var _ = b.Group($"layers.{il}");
                string p = $"layers.{il}.";
                var residual = x;
                var h = b.RmsNorm(x, b.Weight(p + "input_norm.weight"), hp.Eps, $"{p}attn_norm");

                Value attnOut = IsRecurrent(il)
                    ? LinearAttention(b, h, p, b.State(convState[il]), b.State(ssmState[il]))
                    : FullAttention(b, h, p, positions, b.State(kCache[il]), b.State(vCache[il]));

                x = b.Add(attnOut, residual, $"{p}attn_residual");
                var ffnResidual = x;
                h = b.RmsNorm(x, b.Weight(p + "post_attention_norm.weight"), hp.Eps, $"{p}post_attn_norm");
                x = b.Add(FeedForward(b, h, p), ffnResidual, $"{p}l_out");
            }

            using (b.Group("output"))
            {
                var hidden = b.RmsNorm(x, b.Weight("output_norm.weight"), hp.Eps, "result_norm");
                var rows = b.Gather(hidden, outRows, "selected_rows");
                var head = b.TryWeight("lm_head.weight") ?? b.Weight("token_embedding.weight");
                b.Output("logits", b.MatMul(rows, head, "logits"));
                b.Output("hidden", hidden);
            }
            return b.Build();
        }

        Value FullAttention(GraphBuilder b, Value h, string p, Value positions, Value kc, Value vc)
        {
            int D = hp.HeadDim;
            // Joint query + output-gate projection, interleaved per head: [q(D), gate(D)] × heads.
            var qFull = b.Reshape(b.Linear(h, p + "attention.q.weight"), 0, hp.NHead, 2 * D);
            var q = b.Slice(qFull, 2, 0, D);
            var gate = b.Slice(qFull, 2, D, D);
            q = b.RmsNorm(q, b.Weight(p + "attention.q_norm.weight"), hp.Eps);
            var k = b.Reshape(b.Linear(h, p + "attention.k.weight"), 0, hp.NHeadKv, D);
            k = b.RmsNorm(k, b.Weight(p + "attention.k_norm.weight"), hp.Eps);
            var v = b.Reshape(b.Linear(h, p + "attention.v.weight"), 0, hp.NHeadKv, D);

            var rope = hp.MRope
                ? Attributes.From(("dims", hp.RopeDims), ("base", hp.RopeBase), ("style", "neox"), ("sections", hp.RopeSections), ("interleaved", true))
                : Attributes.From(("dims", hp.RopeDims), ("base", hp.RopeBase), ("style", "neox"));
            q = b.Emit1("Rope", [q, positions], rope);
            k = b.Emit1("Rope", [k, positions], rope);

            b.CacheWrite(kc, k);
            b.CacheWrite(vc, v);
            var attn = b.CachedAttention(q, kc, vc, 1f / MathF.Sqrt(D), name: $"{p}attn");
            attn = b.Mul(attn, b.Sigmoid(gate), $"{p}attn_gated");
            return b.Linear(b.Reshape(attn, 0, hp.NHead * D), p + "attention.o.weight", name: $"{p}attn_out");
        }

        // Routed experts (+ shared expert) when the layer has a router, the dense SwiGLU MLP otherwise.
        Value FeedForward(GraphBuilder b, Value h, string p)
        {
            if (MoeBlock.IsPresent(b, p + "mlp."))
            {
                return MoeBlock.Build(b, h, p + "mlp.", hp.Moe, hp.NEmbd, t => b.Silu(t));
            }

            var gate = b.Linear(h, p + "mlp.gate.weight");
            var up = b.Linear(h, p + "mlp.up.weight");
            return b.Linear(b.Mul(b.Silu(gate), up), p + "mlp.down.weight", name: $"{p}ffn_out");
        }

        Value LinearAttention(GraphBuilder b, Value h, string p, Value convSt, Value ssmSt)
        {
            int dk = hp.SsmState;
            int hk = hp.SsmGroups;
            int hv = hp.SsmDtRank;
            int dv = headVDim;
            var qkv = b.Linear(h, p + "linear_attention.qkv.weight");                // [T, 2*hk*dk + hv*dv]
            var z = b.Linear(h, p + "linear_attention.z.weight");                 // [T, hv*dv]
            var beta = b.Sigmoid(b.Linear(h, p + "linear_attention.beta.weight"));    // [T, hv]
            var alpha = b.Linear(h, p + "linear_attention.alpha.weight");             // [T, hv]
            // g = softplus(alpha + dt_bias) * A,  A = -exp(A_log) stored pre-negated as ssm_a
            var g = b.Mul(b.Softplus(b.Add(alpha, b.Weight(p + "linear_attention.dt_bias"))), b.Weight(p + "linear_attention.a"));

            var conv = b.Emit("CausalConv1d", [qkv, b.Weight(p + "linear_attention.conv1d.weight"), convSt],
                Attributes.From(("activation", "silu")), writes: [convSt])[0];
            var q = b.Reshape(b.Slice(conv, 1, 0, hk * dk), 0, hk, dk);
            var k = b.Reshape(b.Slice(conv, 1, hk * dk, hk * dk), 0, hk, dk);
            var v = b.Reshape(b.Slice(conv, 1, 2 * hk * dk, hv * dv), 0, hv, dv);
            q = b.L2Norm(q, hp.Eps);
            k = b.L2Norm(k, hp.Eps);
            if (hp.GroupedValueHeads && hv != hk)
            {
                // Value heads grouped by key head (head h reads key head h / (hv/hk)): repeat each q/k head in place so the
                // recurrence sees one q/k head per value head.
                q = RepeatHeadsInPlace(b, q, hk, hv / hk, dk);
                k = RepeatHeadsInPlace(b, k, hk, hv / hk, dk);
            }

            var o = b.Emit("GatedDeltaRule", [q, k, v, g, beta, ssmSt], Attributes.From(("scale", 1f / MathF.Sqrt(dk))), $"{p}delta", writes: [ssmSt])[0];
            var on = b.RmsNorm(o, b.Weight(p + "linear_attention.norm.weight"), hp.Eps);
            on = b.Mul(on, b.Silu(b.Reshape(z, 0, hv, dv)), $"{p}gated_norm");
            return b.Linear(b.Reshape(on, 0, hv * dv), p + "linear_attention.out.weight", name: $"{p}linear_attn_out");
        }
    }
}
