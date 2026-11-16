using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;
using Sharplight.Vision;

namespace Sharplight.Architectures;

/// <summary>
/// Qwen3.5 / Qwen3.6 / Qwen3.8 hybrid architecture, dense and MoE, and Qwen3-Next (canonical arch <c>qwen3_5</c>):
/// every <c>full_attention_interval</c>-th layer is gated softmax attention with partial (interleaved M-)RoPE; all others are
/// Gated DeltaNet linear-attention layers with a short causal conv. The feed-forward block is a dense SwiGLU MLP or routed
/// experts plus a sigmoid-gated shared expert (<see cref="MoeBlock"/>). MTP layers, when present, form the
/// <see cref="WellKnown.DraftGraph"/> (<see cref="MtpDraftHead"/>, hidden states after the final norm, chained after
/// <c>mtp.output_norm</c>), each a gated-attention layer with its own KV cache; with hyper-connections the head reads the trunk's
/// residual streams and mixes its own (<see cref="MtpDraftHead.BuildHyperConnected"/>). With a span scoring head (<c>span_head.type</c>, e.g. Cloudflare
/// Clef) the trunk also forms the stateful <see cref="WellKnown.ScoreSpansGraph"/> (<see cref="SpanScoring"/>).
/// </summary>
/// <remarks>
/// Canonical description consumed (importers translate their formats into exactly this):
/// <list type="bullet">
/// <item>Hyper-parameters (<c>qwen3_5.*</c>): <c>layer_count</c> (trunk, without MTP), <c>hidden_size</c>,
/// <c>ffn_hidden_size</c>, <c>context_length</c>, <c>norm_eps</c>, <c>attention.head_count</c>,
/// <c>attention.kv_head_count</c>, <c>attention.head_dim</c>, <c>rope.dim</c>, <c>rope.base</c>,
/// <c>rope.sections</c> (4 M-RoPE axes; absent: plain RoPE over one position axis), <c>full_attention_interval</c>,
/// <c>linear_attention.conv_kernel</c>, <c>linear_attention.key_head_count</c>, <c>linear_attention.key_head_dim</c>,
/// <c>linear_attention.value_head_count</c>, <c>linear_attention.value_head_dim</c>,
/// <c>linear_attention.grouped_value_heads</c> (bool, default false), <c>mtp.layer_count</c>; MoE models
/// <c>moe.expert_count</c>, <c>moe.expert_used_count</c>, <c>moe.normalize_weights</c> (default true) and the other
/// <see cref="MoeOptions"/> keys.</item>
/// <item>Weights: <c>token_embedding.weight</c>, <c>output_norm.weight</c>, <c>lm_head.weight</c> (optional, tied
/// to the embedding otherwise); per layer <c>layers.{i}.</c>: <c>input_norm</c>, <c>post_attention_norm</c>,
/// <c>mlp.{gate,up,down}</c> or the MoE weights documented on <see cref="MoeBlock"/> below <c>mlp.</c>; attention layers <c>attention.{q,k,v,o,q_norm,k_norm}</c> (q rows interleave
/// [query | output gate] per head); linear-attention layers <c>linear_attention.{qkv,z,beta,alpha,out,norm,conv1d}</c>
/// (all <c>.weight</c>), <c>linear_attention.a</c> (A = -exp(A_log)) and <c>linear_attention.dt_bias</c>.
/// MTP: <c>mtp.layers.{j}.*</c> (same per-layer names), <c>mtp.{fc,embedding_norm,hidden_norm,output_norm}.weight</c>; with
/// hyper-connections <c>mtp.hidden_norm.weight</c> is per stream [streams·H], <c>mtp.output_hc.*</c> replaces <c>mtp.output_norm</c> and
/// the input projection may be the pair <c>mtp.fc_embedding.weight</c> / <c>mtp.fc_hidden.weight</c>. Files without them
/// (e.g. GGUFs whose converter strips the NextN layers) take them from an auxiliary <see cref="AuxiliaryRoles.Draft"/> source of the same
/// architecture that holds only the MTP layers (an "MTP sidecar", such as unsloth's <c>MTP/mtp-*.gguf</c>); its <c>mtp.*</c> weights and
/// <c>mtp.layer_count</c> are used, everything else (embedding, LM head) comes from the primary source. Such a sidecar on its own is not a model.</item>
/// <item>Semantics: RMS norm weights multiply directly (any "1 + w" convention already folded in); value heads in
/// tiled order (value head h reads key head h mod key_head_count), or grouped by key head (h / (value/key heads)) when
/// <c>linear_attention.grouped_value_heads</c> is set (Qwen3-Next files whose out projection cannot be reordered); conv1d
/// weights [channels, kernel] f32.</item>
/// <item>Qwen3.8-Flash-Next extensions, all optional: <c>hyper_connections.count</c> (residual streams; then the layers have
/// no <c>input_norm</c> / <c>post_attention_norm</c> but the mixers <c>layers.{i}.attention_hc.*</c> / <c>layers.{i}.ffn_hc.*</c>
/// and <c>output_hc.*</c> replaces <c>output_norm</c>, see <see cref="HyperConnections"/>); <c>ngram_embedding.*</c> with
/// <c>layers.{i}.ngram.*</c> (see <see cref="NGramOptions"/> and <see cref="NGramEmbedding"/>);
/// <c>linear_attention.gate_activation</c> ("silu" default, or "sigmoid"); <c>attention.indexer.{head_count, head_dim,
/// block_size, token_budget}</c> with <c>layers.{i}.attention.indexer.{q,k,q_norm,k_norm}.weight</c> (QSA block selection: the
/// raw indexer keys are cached, each complete block of <c>block_size</c> keys is scored by its mean key, RMS-normed and rotated at
/// the block start, and the attention sees the best <c>token_budget / block_size</c> blocks plus the incomplete last block, see
/// <see cref="IndexerSelect"/>).</item>
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
    private const string MtpPrefix = "mtp.";
    private static readonly IMlpActivation s_silu = MlpActivations.Create("silu");

    /// <summary>A source with MTP layers but no trunk layers: the NextN layers of a model stored on their own.</summary>
    private static bool IsMtpSidecar(ModelSource source) => MtpDraftHead.IsSidecar(source);

    /// <summary>
    /// The MTP sidecar to take the draft layers from: the auxiliary draft source when the primary has no MTP layers of its own and the draft source
    /// is a sidecar of the same architecture and width; null otherwise (a draft source of another kind is not this adapter's).
    /// </summary>
    private static ModelSource? MtpSidecar(ModelSourceSet sources)
    {
        var src = sources.Primary;
        if (sources.Auxiliary.GetValueOrDefault(AuxiliaryRoles.Draft) is not { } draft || !IsMtpSidecar(draft)
            || src.Weights.Keys.Any(k => k.StartsWith(MtpPrefix, StringComparison.Ordinal)))
        {
            return null;
        }

        if (!string.Equals(draft.Architecture, src.Architecture, StringComparison.OrdinalIgnoreCase)
            || draft.Arch<int>("hidden_size") != src.Arch<int>("hidden_size"))
        {
            throw new NotSupportedException($"The MTP draft file '{draft.Path}' ({draft.Architecture}, width {draft.Arch<int>("hidden_size")}) does not belong to "
                + $"'{src.Path}' ({src.Architecture}, width {src.Arch<int>("hidden_size")}).");
        }

        return draft;
    }

    /// <inheritdoc/>
    public string Name => "Qwen3.5 hybrid (Gated DeltaNet + gated attention)";
    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => ["qwen3_5"];

    private sealed record HParams(
        int NLayer, int NEmbd, int NFf, int NHead, int NHeadKv, int HeadDim, float Eps,
        int RopeDims, float RopeBase, int[] RopeSections,
        int ConvKernel, int SsmState, int SsmGroups, int SsmDtRank, int SsmInner, int FullAttnInterval,
        int Vocab, int ContextLength, bool MRope, bool GroupedValueHeads, MoeOptions Moe,
        int Streams, NGramOptions? NGram, bool SigmoidOutputGate);

    private const int MRopeAxes = 4;

    // [T, heads, dim] → [T, heads·repeats, dim] with each head repeated <repeats> times consecutively.
    private static Value RepeatHeadsInPlace(GraphBuilder b, Value x, int heads, int repeats, int dim)
    {
        var tiled = b.Emit1("Tile", [b.Reshape(x, 0, heads, 1, dim)], Attributes.From(("repeats", new long[] { 1, 1, repeats, 1 })));
        return b.Reshape(tiled, 0, (long)heads * repeats, dim);
    }

    /// <summary>
    /// QSA block selection of one attention layer: caches the raw indexer key of every row and returns the positions each query row
    /// attends, i32 [T, budget + block − 1] (IndexerTopK, group score "mean"). Queries are the RMS-normed indexer query heads rotated
    /// like the attention queries; a block's key is the mean of its raw keys, RMS-normed with <c>k_norm</c> and rotated at the block's
    /// first position; its score is <c>Σ_h relu(q_h · key) / sqrt(D)</c>.
    /// </summary>
    private static Value IndexerSelect(GraphBuilder b, Value h, string p, Value positions, StateDefinition keyCache, Attributes rope,
        int ropeDims, float ropeBase, float eps, int budget, int block)
    {
        int dims = (int)b.Weight(p + "attention.indexer.k.weight").Shape[0].Fixed;
        int heads = (int)(b.Weight(p + "attention.indexer.q.weight").Shape[0].Fixed / dims);
        var q = b.Reshape(b.Linear(h, p + "attention.indexer.q.weight", name: $"{p}indexer_q"), 0, heads, dims);
        q = b.Emit1("Rope", [b.RmsNorm(q, b.Weight(p + "attention.indexer.q_norm.weight"), eps), positions], rope);
        var keys = b.State(keyCache);
        b.CacheWrite(keys, b.Reshape(b.Linear(h, p + "attention.indexer.k.weight", name: $"{p}indexer_k"), 0, 1, dims));

        // Unweighted head sum: every head weighs 1/sqrt(D) (a [T, heads] constant derived from q so it follows the row count).
        var weights = b.Scale(b.Reshape(b.Slice(q, 2, 0, 1), 0, heads), 0f, 1f / MathF.Sqrt(dims), $"{p}indexer_w");
        b.RuntimeScalar(WellKnown.PastScalar);
        var attributes = Attributes.From(("k", (long)budget), ("offset", WellKnown.PastScalar), ("group", (long)block), ("group_score", "mean"),
            ("tail", true), ("pool_norm_eps", eps), ("pool_rope_dims", (long)ropeDims), ("pool_rope_base", ropeBase));
        return b.Emit1("IndexerTopK", [q, keys, weights, b.Weight(p + "attention.indexer.k_norm.weight")], attributes, $"{p}indexer_topk");
    }

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        if (IsMtpSidecar(src))
        {
            throw MtpDraftHead.SidecarAsModelError(src);
        }

        var sidecar = MtpSidecar(sources);
        var mtpSource = sidecar ?? src;
        int nextn = mtpSource.Arch("mtp.layer_count", 0);
        if (nextn == 0)
        {
            // Checkpoints whose config leaves out the MTP layer count (tiny Qwen3-Next): the layers the weights carry.
            nextn = mtpSource.Weights.Keys.Where(k => k.StartsWith("mtp.layers.", StringComparison.Ordinal))
                .Select(k => k.Split('.')[2]).Distinct(StringComparer.Ordinal).Count();
        }
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
            Moe: MoeOptions.Read(src, normalizeDefault: true),
            Streams: src.Arch("hyper_connections.count", 0),
            NGram: NGramOptions.Read(src),
            SigmoidOutputGate: src.Arch("linear_attention.gate_activation", "silu") switch
            {
                "silu" => false,
                "sigmoid" => true,
                var other => throw new InvalidDataException($"Unknown linear_attention.gate_activation '{other}'."),
            });
        if (hp.NGram is not null && hp.Streams == 0)
        {
            throw new InvalidDataException("The n-gram embedding needs hyper-connections (hyper_connections.count).");
        }

        bool IsRecurrent(int il) => (il + 1) % hp.FullAttnInterval != 0;
        int positionAxes = hp.MRope ? MRopeAxes : 1;

        // ---- persistent state ---------------------------------------------------------------
        var states = new List<StateDefinition>();
        var kCache = new Dictionary<int, StateDefinition>();
        var vCache = new Dictionary<int, StateDefinition>();
        var indexerCache = new Dictionary<int, StateDefinition>();
        int indexerBudget = src.Arch("attention.indexer.token_budget", 0);
        int indexerBlock = src.Arch("attention.indexer.block_size", 1);
        var convState = new Dictionary<int, StateDefinition>();
        var ssmState = new Dictionary<int, StateDefinition>();
        var ngramStates = new Dictionary<int, (StateDefinition Window, StateDefinition Conv)>();
        int convChannels = hp.SsmInner + 2 * hp.SsmGroups * hp.SsmState;
        int headVDim = hp.SsmInner / hp.SsmDtRank;
        for (int il = 0; il < hp.NLayer; il++)
        {
            string g = $"layers.{il}";
            if (IsRecurrent(il))
            {
                states.Add(convState[il] = RecurrentStates.ConvWindow(g, hp.ConvKernel - 1, convChannels));
                states.Add(ssmState[il] = RecurrentStates.Scan(g, hp.SsmDtRank, headVDim, hp.SsmState, options.RecurrentStateType));
            }
            else
            {
                states.Add(kCache[il] = new StateDefinition { Name = $"{g}.k_cache", Type = options.KvCacheTypeFor(hp.HeadDim), Shape = new Shape(Dim.Sym(WellKnown.ContextSymbol), hp.NHeadKv, hp.HeadDim), Kind = StateKind.KeyCache, Group = g });
                states.Add(vCache[il] = new StateDefinition { Name = $"{g}.v_cache", Type = options.KvCacheTypeFor(hp.HeadDim), Shape = new Shape(Dim.Sym(WellKnown.ContextSymbol), hp.NHeadKv, hp.HeadDim), Kind = StateKind.ValueCache, Group = g });
                if (indexerBudget > 0 && src.Weight($"{g}.attention.indexer.k.weight") is { } indexerKeys)
                {
                    states.Add(indexerCache[il] = IndexerKeyCache(g, indexerKeys));
                }
            }

            if (hp.NGram?.Layer == il)
            {
                var ngram = ngramStates[il] = NGramEmbedding.States(g, hp.NGram, src.Weight($"{g}.ngram.conv1d.weight")!);
                states.Add(ngram.Window);
                states.Add(ngram.Conv);
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
        bool mergeVision = vision is not null && !ReferenceEquals(visionSrc, src);
        if (mergeVision || hp.NGram is not null || sidecar is not null)
        {
            var merged = new Dictionary<string, WeightInfo>(src.Weights, StringComparer.Ordinal);
            if (mergeVision)
            {
                foreach (var (k, w) in visionSrc.Weights)
                {
                    merged[k] = w;
                }
            }

            if (sidecar is not null)
            {
                foreach (var (k, w) in sidecar.Weights)
                {
                    if (k.StartsWith(MtpPrefix, StringComparison.Ordinal))
                    {
                        merged[k] = w;
                    }
                }
            }

            if (hp.NGram is not null)
            {
                NGramEmbedding.AddDerivedWeights(merged, hp.NGram);
            }

            allWeights = merged;
        }
        WeightInfo? Lookup(string n) => allWeights.TryGetValue(n, out var w) ? w : null;
        // MTP draft head; with hyper-connections it reads the trunk's residual streams (HyperConnections.StreamsOutput).
        bool headPresent = hp.Streams > 0 ? MtpDraftHead.IsHyperConnectedPresent(Lookup) : MtpDraftHead.IsPresent(Lookup);
        int draftLayers = nextn > 0 && headPresent ? nextn : 0;
        var graphs = new Dictionary<string, ComputeGraph>
        {
            [WellKnown.DecodeGraph] = BuildTrunk(WellKnown.DecodeGraph, embeddingsInput: false),
        };
        if (draftLayers > 0)
        {
            var mtpK = new StateDefinition[draftLayers];
            var mtpV = new StateDefinition[draftLayers];
            var mtpIndexer = new StateDefinition?[draftLayers];
            for (int j = 0; j < draftLayers; j++)
            {
                string g = $"mtp.layers.{j}";
                states.Add(mtpK[j] = new StateDefinition { Name = $"{g}.k_cache", Type = options.KvCacheTypeFor(hp.HeadDim), Shape = new Shape(Dim.Sym(WellKnown.ContextSymbol), hp.NHeadKv, hp.HeadDim), Kind = StateKind.KeyCache, Group = g });
                states.Add(mtpV[j] = new StateDefinition { Name = $"{g}.v_cache", Type = options.KvCacheTypeFor(hp.HeadDim), Shape = new Shape(Dim.Sym(WellKnown.ContextSymbol), hp.NHeadKv, hp.HeadDim), Kind = StateKind.ValueCache, Group = g });
                if (indexerBudget > 0 && Lookup($"{g}.attention.indexer.k.weight") is { } indexerKeys)
                {
                    var indexerState = IndexerKeyCache(g, indexerKeys);
                    mtpIndexer[j] = indexerState;
                    states.Add(indexerState);
                }
            }

            graphs[WellKnown.DraftGraph] = hp.Streams > 0
                ? MtpDraftHead.BuildHyperConnected(Lookup, hp.NEmbd, hp.Streams, positionAxes, hp.Eps, draftLayers, (b, x, j, positions) =>
                {
                    string p = $"mtp.layers.{j}.";
                    var (h, inject) = HyperConnections.Mix(b, x, p + "attention_hc.", hp.Streams, hp.NEmbd, hp.Eps);
                    var attnOut = FullAttention(b, h, p, positions, b.State(mtpK[j]), b.State(mtpV[j]), mtpIndexer[j]);
                    x = HyperConnections.Inject(b, x, attnOut, inject!, hp.Streams, hp.NEmbd, $"{p}attn_residual");
                    (h, inject) = HyperConnections.Mix(b, x, p + "ffn_hc.", hp.Streams, hp.NEmbd, hp.Eps);
                    return HyperConnections.Inject(b, x, FeedForward(b, h, p), inject!, hp.Streams, hp.NEmbd, $"{p}l_out");
                })
                : MtpDraftHead.Build(Lookup, hp.NEmbd, positionAxes, hp.Eps, draftLayers, (b, x, j, positions) =>
            {
                string p = $"mtp.layers.{j}.";
                var residual = x;
                var h = b.RmsNorm(x, b.Weight(p + "input_norm.weight"), hp.Eps, $"{p}attn_norm");
                x = b.Add(FullAttention(b, h, p, positions, b.State(mtpK[j]), b.State(mtpV[j])), residual, $"{p}attn_residual");
                var ffnResidual = x;
                h = b.RmsNorm(x, b.Weight(p + "post_attention_norm.weight"), hp.Eps, $"{p}post_attn_norm");
                return b.Add(FeedForward(b, h, p), ffnResidual, $"{p}l_out");
            }, chainNormalized: true);
        }

        // The n-gram embedding hashes token ids, so a graph fed with embeddings only cannot run it.
        if (options.IncludeAuxiliaryGraphs && hp.NGram is null)
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
            [WellKnown.DraftLayers] = draftLayers,
        };
        if (hp.Streams > 0 && draftLayers > 0)
        {
            props[WellKnown.DraftHiddenSource] = HyperConnections.StreamsOutput;
        }

        // Decision models (Clef): the trunk run from reset states over one prompt, then the span scoring head.
        if (SpanScoring.IsPresent(src))
        {
            graphs[WellKnown.ScoreSpansGraph] = BuildTrunk(WellKnown.ScoreSpansGraph, embeddingsInput: false, scoreSpans: true);
            props[WellKnown.SpanHeadType] = SpanScoring.HeadType(src);
        }

        if (vision is not null)
        {
            if (vision.DeepstackLayers.Length > 0)
            {
                throw new NotSupportedException("qwen3_5: vision towers with DeepStack layers belong to the decoder architecture (Qwen3-VL).");
            }

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

        ComputeGraph BuildTrunk(string name, bool embeddingsInput, bool mixed = false, bool scoreSpans = false)
        {
            var b = new GraphBuilder(name, Lookup);
            var T = Dim.Sym(WellKnown.TokenSymbol);
            Value x;
            Value? tokens = null;
            using (b.Group("input"))
            {
                if (mixed)
                {
                    // Rows are token embeddings or media embeddings: x = tok·(1-mask) + emb·mask.
                    tokens = b.Input("tokens", InputRole.TokenIds, DataType.I32, new Shape(T));
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
                    tokens = b.Input("tokens", InputRole.TokenIds, DataType.I32, new Shape(T));
                    x = b.Gather(b.Weight("token_embedding.weight"), tokens, "inp_embd");
                }
            }
            var positions = b.Input("positions", InputRole.Positions, DataType.I32, new Shape(T, positionAxes));
            Value? outRows = null, sequenceSpans = null;
            SpanScoringInputs? spans = null;
            if (scoreSpans)
            {
                sequenceSpans = b.Input(PackedSequences.SequenceSpansInput, InputRole.SequenceSpans, DataType.I32, new Shape(Dim.Sym(WellKnown.SequenceSymbol), 2));
                spans = SpanScoring.Declare(b);
            }
            else
            {
                outRows = b.Input("output_rows", InputRole.OutputRows, DataType.I32, new Shape(Dim.Sym("R")));
            }

            if (hp.Streams > 0)
            {
                x = HyperConnections.Expand(b, x, hp.Streams);
            }

            for (int il = 0; il < hp.NLayer; il++)
            {
                using var _ = b.Group($"layers.{il}");
                string p = $"layers.{il}.";
                if (hp.Streams > 0)
                {
                    x = HyperConnectedLayer(b, x, il, positions, tokens);
                    continue;
                }

                var residual = x;
                var h = b.RmsNorm(x, b.Weight(p + "input_norm.weight"), hp.Eps, $"{p}attn_norm");

                Value attnOut = IsRecurrent(il)
                    ? LinearAttention(b, h, p, b.State(convState[il]), b.State(ssmState[il]))
                    : FullAttention(b, h, p, positions, b.State(kCache[il]), b.State(vCache[il]), indexerCache.GetValueOrDefault(il));

                x = b.Add(attnOut, residual, $"{p}attn_residual");
                var ffnResidual = x;
                h = b.RmsNorm(x, b.Weight(p + "post_attention_norm.weight"), hp.Eps, $"{p}post_attn_norm");
                x = b.Add(FeedForward(b, h, p), ffnResidual, $"{p}l_out");
            }

            using (b.Group("output"))
            {
                var hidden = hp.Streams > 0
                    ? HyperConnections.Mix(b, x, "output_hc.", hp.Streams, hp.NEmbd, hp.Eps).Mixed
                    : b.RmsNorm(x, b.Weight("output_norm.weight"), hp.Eps, "result_norm");
                string headName = Lookup("lm_head.weight") is null ? "token_embedding.weight" : "lm_head.weight";
                if (scoreSpans)
                {
                    SpanScoring.Output(b, src, hidden, tokens!, sequenceSpans!, spans!, headName);
                    return b.Build();
                }

                var rows = b.Gather(hidden, outRows!, "selected_rows");
                b.Output("logits", b.MatMul(rows, b.Weight(headName), "logits"));
                b.Output("hidden", hidden);
                if (hp.Streams > 0 && draftLayers > 0)
                {
                    b.Output(HyperConnections.StreamsOutput, x);
                }
            }
            return b.Build();
        }

        // Raw (un-normed, unrotated) indexer keys [ctx, 1, D] of an attention layer: blocks are pooled and normed when scored.
        StateDefinition IndexerKeyCache(string group, WeightInfo indexerKeys)
        {
            var kvType = options.KvCacheTypeFor(hp.HeadDim);
            return new StateDefinition
            {
                Name = $"{group}.indexer_key_cache",
                Type = kvType == DataType.F32 ? DataType.F32 : DataType.F16,
                Shape = new Shape(Dim.Sym(WellKnown.ContextSymbol), 1, indexerKeys.Shape[0].Fixed),
                Kind = StateKind.KeyCache,
                Group = group,
            };
        }

        // Hyper-connected layer: optional n-gram features, then attention and feed-forward each between a stream mixer and an
        // injection back into the streams (no separate norms, no plain residual adds).
        Value HyperConnectedLayer(GraphBuilder b, Value x, int il, Value positions, Value? tokens)
        {
            string p = $"layers.{il}.";
            if (ngramStates.TryGetValue(il, out var ngram))
            {
                var features = NGramEmbedding.Build(b, tokens!, x, p, hp.NGram!, hp.Streams, hp.NEmbd, hp.Eps, b.State(ngram.Window), b.State(ngram.Conv));
                x = b.Add(x, features, $"{p}ngram_residual");
            }

            var (h, inject) = HyperConnections.Mix(b, x, p + "attention_hc.", hp.Streams, hp.NEmbd, hp.Eps);
            Value attnOut = IsRecurrent(il)
                ? LinearAttention(b, h, p, b.State(convState[il]), b.State(ssmState[il]))
                : FullAttention(b, h, p, positions, b.State(kCache[il]), b.State(vCache[il]), indexerCache.GetValueOrDefault(il));
            x = HyperConnections.Inject(b, x, attnOut, inject!, hp.Streams, hp.NEmbd, $"{p}attn_residual");

            (h, inject) = HyperConnections.Mix(b, x, p + "ffn_hc.", hp.Streams, hp.NEmbd, hp.Eps);
            return HyperConnections.Inject(b, x, FeedForward(b, h, p), inject!, hp.Streams, hp.NEmbd, $"{p}l_out");
        }

        Value FullAttention(GraphBuilder b, Value h, string p, Value positions, Value kc, Value vc, StateDefinition? indexerKeys = null)
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
            Value attn;
            if (indexerKeys is null)
            {
                attn = b.CachedAttention(q, kc, vc, 1f / MathF.Sqrt(D), name: $"{p}attn");
            }
            else
            {
                var selection = IndexerSelect(b, h, p, positions, indexerKeys, rope, hp.RopeDims, hp.RopeBase, hp.Eps, indexerBudget, indexerBlock);
                b.RuntimeScalar(WellKnown.PastScalar);
                attn = b.Emit1("CachedAttention", [q, kc, vc, selection],
                    Attributes.From(("scale", 1f / MathF.Sqrt(D)), ("offset", WellKnown.PastScalar), ("selection", true)), $"{p}attn");
            }

            attn = b.Mul(attn, b.Sigmoid(gate), $"{p}attn_gated");
            return b.Linear(b.Reshape(attn, 0, hp.NHead * D), p + "attention.o.weight", name: $"{p}attn_out");
        }

        // Routed experts (+ shared expert) when the layer has a router, the dense SwiGLU MLP otherwise.
        Value FeedForward(GraphBuilder b, Value h, string p)
        {
            if (MoeBlock.IsPresent(b, p + "mlp."))
            {
                return MoeBlock.Build(b, h, p, "mlp.", hp.Moe, hp.NEmbd, s_silu);
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
            var zh = b.Reshape(z, 0, hv, dv);
            on = b.Mul(on, hp.SigmoidOutputGate ? b.Sigmoid(zh) : b.Silu(zh), $"{p}gated_norm");
            return b.Linear(b.Reshape(on, 0, hv * dv), p + "linear_attention.out.weight", name: $"{p}linear_attn_out");
        }
    }
}
