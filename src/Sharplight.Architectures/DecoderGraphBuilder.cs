using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Emits the compute graphs of the canonical <c>decoder</c> architecture (see <see cref="DecoderAdapter"/> for the weight and
/// metadata vocabulary): the cached-attention decode graphs and the packed-sequence embedding and classification graphs.
/// Partial files: attention (<c>DecoderGraphBuilder.Attention.cs</c>) and feed-forward blocks (<c>DecoderGraphBuilder.Mlp.cs</c>).
/// </summary>
internal sealed partial class DecoderGraphBuilder
{
    private readonly ModelSource _src;
    private readonly AdapterOptions _options;
    private readonly DecoderHyperParameters _hp;
    private readonly bool _longRope;
    private readonly DecoderNorm _norm;
    private readonly IMlpActivation _activation;
    private readonly StateDefinition?[] _kCache;
    private readonly StateDefinition?[] _vCache;
    private readonly BlockSparseIndexer? _blockIndexer;          // only when the context exceeds its budget (below it every key is attended)
    private readonly StateDefinition?[] _blockIndexerCache;      // per cache slot: the block indexer's key cache of a sparse layer
    private Func<string, WeightInfo?>? _draft;
    private readonly int[] _taps;                                 // layers whose inputs the decode graphs export (an EAGLE-3 drafter's features)
    private Dictionary<int, Value>? _tapValues;                   // while a trunk graph is built: the tapped layer inputs
    private Value? _mediaMask;                                     // while the multimodal graph is built: the per-row media weight-set mask
    private Value? _valueEmbeddings;                              // while a graph with token inputs is built: value_embedding rows [T, KvH·D]
    private readonly int[] _kvExports;                            // layers whose keys and values the decode graphs export (a shared-KV drafter's)
    private Dictionary<int, (Value K, Value V)>? _kvExportValues; // while a trunk graph is built: the exported keys and values by layer
    private Dictionary<int, (Value K, Value V)>? _suppliedKv;     // while a shared-KV draft graph is built: keys and values from the input, by slot
    private readonly (StateDefinition Key, StateDefinition Value)? _routerCache; // token-activated adapter routing (AdapterRouter): its KV cache
    private Value? _adapterSlots;                                 // while a graph with adapter routing is built: every row's adapter slot, i32 [T, 1]
    private string _pooling = "last";
    private bool _normalize = true;

    /// <summary>Creates the builder and the KV-cache state definitions.</summary>
    /// <param name="source">The model source.</param>
    /// <param name="adapterOptions">Adapter options.</param>
    /// <param name="hyperParameters">The hyper-parameters.</param>
    /// <param name="taps">
    /// Layers whose inputs the token decode graphs also output, concatenated, as <see cref="Eagle3DraftHead.TapsOutput"/> (the layer count: the
    /// last layer's output); empty for none.
    /// </param>
    /// <param name="kvExports">
    /// Layers whose keys and values (as written to their KV cache) the token decode graphs also output, after the final hidden state, as
    /// <see cref="SharedKvDraftHead.FeaturesOutput"/>; null for none.
    /// </param>
    /// <param name="statePrefix">Prefix of the KV-cache state names and groups (a drafter's caches next to its target's).</param>
    /// <exception cref="InvalidDataException">A KV-sharing layer does not point to an earlier layer.</exception>
    public DecoderGraphBuilder(ModelSource source, AdapterOptions adapterOptions, DecoderHyperParameters hyperParameters, int[]? taps = null,
        int[]? kvExports = null, string statePrefix = "")
    {
        _taps = taps ?? [];
        _kvExports = kvExports ?? [];
        if (_taps.Any(il => il < 0 || il > hyperParameters.LogicalLayers) || (_taps.Length > 0 && (hyperParameters.AltUpStreams > 0 || hyperParameters.Cycles is not null)))
        {
            throw new InvalidDataException($"decoder: cannot export the inputs of layers {string.Join(", ", _taps)}.");
        }

        _src = source;
        _options = adapterOptions;
        _hp = hyperParameters;
        _norm = new DecoderNorm(_hp.NormType, _hp.NormGroups);
        _activation = MlpActivations.Create(_hp);
        int origCtx = _src.Arch("rope.original_context_length", 0);
        int runtimeCtx = _options.ContextLength > 0 ? _options.ContextLength : _hp.ContextLength;
        _longRope = UseLongRope(origCtx, runtimeCtx);
        States = [];
        // A NextN draft head (one MTP layer after the trunk's, see MtpDraftHead.FirstNextnModule) when the layers are uniform: the
        // MTP layer is built like a trunk layer with index NLayer and has its own KV cache.
        _draft = _hp.UniformLayers && _hp.NormType == "rms" && _hp.PreNorm && _hp.Causal ? MtpDraftHead.FirstNextnModule(_src.Weight) : null;
        // Cache slots: one per logical layer (layer loops run the physical layers again with caches of their own), then the draft layer.
        int logical = _hp.LogicalLayers;
        // Bidirectional models (attention.causal = false) only run the packed-sequence graphs: no KV caches.
        int slots = !_hp.Causal ? 0 : logical + (_draft is null ? 0 : 1);
        _kCache = new StateDefinition?[slots];
        _vCache = new StateDefinition?[slots];
        for (int slot = 0; slot < slots; slot++)
        {
            int il = slot < logical ? _hp.PhysicalLayerOf(slot) : _hp.NLayer;
            string g = statePrefix + (slot < logical ? $"layers.{slot}" : $"mtp.layers.{slot - logical}");
            int sourceSlot = slot - il + _hp.KvSourceOf(il);
            if (sourceSlot != slot)
            {
                if (sourceSlot > slot || sourceSlot < 0)
                {
                    throw new InvalidDataException("decoder: attention.kv_source_layers must point to an earlier layer.");
                }

                _kCache[slot] = _kCache[sourceSlot];
                _vCache[slot] = _vCache[sourceSlot];
                continue;
            }

            if (_hp.KvHeadsOf(il) == 0)
            {
                continue;   // no attention, or a linear replacement of it (Deci): no cache
            }

            var keyShape = new Shape(Dim.Sym(WellKnown.ContextSymbol), _hp.KvHeadsOf(il), _hp.HeadDimOf(il));
            var valueShape = new Shape(Dim.Sym(WellKnown.ContextSymbol), _hp.KvHeadsOf(il), _hp.ValueDimOf(il));
            States.Add(_kCache[slot] = new StateDefinition { Name = $"{g}.k_cache", Type = _options.KvCacheTypeFor(_hp.HeadDimOf(il)), Shape = keyShape, Kind = StateKind.KeyCache, Group = g });
            States.Add(_vCache[slot] = new StateDefinition { Name = $"{g}.v_cache", Type = _options.KvCacheTypeFor(_hp.ValueDimOf(il)), Shape = valueShape, Kind = StateKind.ValueCache, Group = g });
        }

        // Block-sparse attention (MiniMax-M3): the layers with a block indexer attend only the key blocks it picks.
        if (BlockSparseIndexerOptions.Read(_src) is { } blockIndexer && runtimeCtx > blockIndexer.TokenBudget)
        {
            _blockIndexer = new BlockSparseIndexer(blockIndexer, _hp.Eps);
        }

        _blockIndexerCache = new StateDefinition?[slots];
        for (int slot = 0; slot < slots && _blockIndexer is not null; slot++)
        {
            string prefix = slot < logical ? $"layers.{_hp.PhysicalLayerOf(slot)}." : "mtp.layers.0.";
            if (_kCache[slot] is not null && BlockSparseIndexer.IsPresent(_src.Weight, prefix))
            {
                string g = slot < logical ? $"layers.{slot}" : "mtp.layers.0";
                States.Add(_blockIndexerCache[slot] = BlockSparseIndexer.CreateKeyCache(_src.Weight, prefix, g, _options.KvCacheType));
            }
        }

        // Token-activated adapters (Granite Switch): the router's single attention head has a cache of its own.
        if (_hp.Causal && AdapterRouter.IsPresent(_src))
        {
            var router = AdapterRouter.CreateCaches(statePrefix + "adapter_router", _options.KvCacheTypeFor(AdapterRouter.HeadDim));
            States.Add(router.Key);
            States.Add(router.Value);
            _routerCache = router;
        }
    }

    /// <summary>The persistent states (KV caches) the graphs use.</summary>
    public List<StateDefinition> States { get; }

    /// <summary>Whether the model has a NextN draft head (<see cref="WellKnown.DraftGraph"/>); valid after <see cref="BuildGraphs"/>.</summary>
    public bool HasDraftHead => _draft is not null;

    /// <summary>Pooling of the packed-sequence embedding graph; valid after <see cref="BuildGraphs"/>.</summary>
    public string Pooling => _pooling;

    /// <summary>Whether the embedding graph L2-normalizes; valid after <see cref="BuildGraphs"/>.</summary>
    public bool Normalize => _normalize;

    // Longrope: the long factors replace the short ones when the configured context exceeds the original training context.
    private static bool UseLongRope(int originalContext, int configuredContext) => originalContext > 0 && configuredContext > originalContext;

    /// <summary>Builds every graph the model supports, keyed by graph name.</summary>
    public Dictionary<string, ComputeGraph> BuildGraphs()
    {
        _pooling = PackedSequences.NormalizePooling(_options.EmbeddingPooling) ?? PackedSequences.NormalizePooling(_src.Get<string?>(CanonicalKeys.EmbeddingPooling, null))
            ?? (_hp.Causal ? "last" : "mean");
        _normalize = _options.EmbeddingNormalize ?? _src.Get(CanonicalKeys.EmbeddingNormalize, true);
        if (!_hp.Causal)
        {
            // Bidirectional encoders with a decoder's layers (EuroBERT, NeoBERT, ModernBERT, EmbeddingGemma, Llama embedding models): every token
            // sees the whole sequence, so there is nothing to decode token by token; the embedding graph is the model.
            var packed = new Dictionary<string, ComputeGraph> { [WellKnown.EmbedGraph] = BuildPacked(WellKnown.EmbedGraph, classify: false) };
            if (IsDiffusion)
            {
                packed[WellKnown.DenoiseGraph] = BuildDenoise();
            }

            return packed;
        }

        var graphs = new Dictionary<string, ComputeGraph> { [WellKnown.DecodeGraph] = BuildTrunk(WellKnown.DecodeGraph, embeddingsInput: false) };
        if (_draft is not null)
        {
            try
            {
                graphs[WellKnown.DraftGraph] = MtpDraftHead.Build(_draft, _hp.NEmbd, _hp.PositionAxes, _hp.Eps, layers: 1, (b, x, j, positions) =>
                {
                    string p = $"mtp.layers.{j}.";
                    var slot = new LayerSlot(_hp.NLayer + j, _hp.LogicalLayers + j);
                    var packedKv = new Dictionary<int, (Value K, Value V)>();
                    return _hp.ParallelResidual
                        ? ParallelLayer(b, x, p, slot, positions, FrequencyFactors(b), null, packedKv, null)
                        : SequentialLayer(b, x, p, slot, positions, FrequencyFactors(b), null, packedKv, null);
                }, chainNormalized: true);
            }
            catch (KeyNotFoundException)
            {
                // The checkpoint's MTP layer lacks weights a trunk layer has (an importer that does not know the family's MTP tensors):
                // no draft head.
                _draft = null;
                States.RemoveAll(s => s.Group == "mtp.layers.0");
            }
        }

        // Value embeddings and adapter routing are looked up by token id: no embeddings-input graph.
        if (_options.IncludeAuxiliaryGraphs && _src.Weight(ValueEmbeddingWeight) is null && _routerCache is null)
        {
            graphs["decode_embeddings"] = BuildTrunk("decode_embeddings", embeddingsInput: true);
        }

        // Packed-sequence graphs (embedding models such as Qwen3-Embedding, yes/no rerankers such as Qwen3-Reranker): causal attention
        // within each sequence, no KV cache. Last-token pooling unless the model declares otherwise. Not built for models with
        // attention sinks (the packed attention has none).
        bool packable = !Enumerable.Range(0, _hp.NLayer).Any(il => _src.Weight($"layers.{il}.attention.sinks") is not null);
        if (_options.IncludeAuxiliaryGraphs && packable)
        {
            graphs[WellKnown.EmbedGraph] = BuildPacked(WellKnown.EmbedGraph, classify: false);
            graphs[WellKnown.ClassifyGraph] = BuildPacked(WellKnown.ClassifyGraph, classify: true);
        }

        return graphs;
    }

    /// <summary>
    /// The <see cref="WellKnown.DecodeMultimodalGraph"/>: token ids plus an embeddings input [T, H·(1 + <paramref name="deepstackLayers"/>)] and
    /// a row mask [T, 1]. Rows with mask 1 take their input from the first H columns instead of the token embedding; the k-th further block
    /// of H columns (DeepStack, zero on text rows) is added to the hidden state after layer k. Layers with a media weight set (<see cref="TokenWeightSets"/>,
    /// CogVLM's visual experts) also take the <see cref="InputRole.MediaWeightMask"/> input; every other graph uses their text weights only.
    /// </summary>
    public ComputeGraph BuildMultimodal(int deepstackLayers) => BuildTrunk(WellKnown.DecodeMultimodalGraph, embeddingsInput: false, deepstackLayers);

    private ComputeGraph BuildTrunk(string name, bool embeddingsInput, int deepstackLayers = -1)
    {
        var b = new GraphBuilder(name, _src.Weight);
        var T = Dim.Sym(WellKnown.TokenSymbol);
        Value x;
        Value positions;
        Value? perLayer = null;
        Value? deepstack = null;
        using (b.Group("input"))
        {
            Value? tokens = null;
            if (embeddingsInput)
            {
                x = b.Input("embeddings", InputRole.Embeddings, DataType.F32, new Shape(T, _hp.NEmbd));
                positions = b.Input("positions", InputRole.Positions, DataType.I32, new Shape(T, _hp.PositionAxes));
            }
            else
            {
                tokens = b.Input("tokens", InputRole.TokenIds, DataType.I32, new Shape(T));
                positions = b.Input("positions", InputRole.Positions, DataType.I32, new Shape(T, _hp.PositionAxes));
                x = TokenEmbeddings(b, RouteAdapters(b, tokens, rowSpans: null));
            }

            if (deepstackLayers >= 0)
            {
                // Media rows: x = tok·(1 − mask) + emb·mask.
                var emb = b.Input("embeddings", InputRole.Embeddings, DataType.F32, new Shape(T, (long)_hp.NEmbd * (1 + deepstackLayers)));
                var mask = b.Input("embedding_mask", InputRole.EmbeddingMask, DataType.F32, new Shape(T, 1));
                var media = deepstackLayers == 0 ? emb : b.Slice(emb, 1, 0, _hp.NEmbd);
                x = b.Add(b.Mul(x, b.Scale(mask, -1f, 1f)), b.Mul(media, mask), "inp_mixed");
                deepstack = deepstackLayers == 0 ? null : b.Slice(emb, 1, _hp.NEmbd, (long)_hp.NEmbd * deepstackLayers);
                if (TokenWeightSets.Present(b, _hp.NLayer, "attention.q", "attention.k", "attention.v", "attention.o", "mlp.up", "mlp.gate", "mlp.down"))
                {
                    _mediaMask = TokenWeightSets.DeclareMask(b, T);
                }
            }

            x = AddPositionEmbeddings(b, x, positions);
            if (_hp.PerLayerSize > 0)
            {
                perLayer = PerLayerInputs(b, x, tokens);
            }
        }

        var outRows = b.Input("output_rows", InputRole.OutputRows, DataType.I32, new Shape(Dim.Sym("R")));
        _tapValues = _taps.Length > 0 ? [] : null;
        _kvExportValues = _kvExports.Length > 0 ? [] : null;
        x = Layers(b, x, positions, FrequencyFactors(b), perLayer, rowSpans: null, deepstack);
        if (_tapValues is not null)
        {
            _tapValues[_hp.LogicalLayers] = x;
        }

        using (b.Group("output"))
        {
            var hidden = OutputNorm(b, x);
            var rows = b.Gather(hidden, outRows, "selected_rows");
            b.Output("logits", Logits(b, rows));
            b.Output("hidden", hidden);
            if (_tapValues is not null)
            {
                b.Output(Eagle3DraftHead.TapsOutput, b.Concat(1, [.. _taps.Select(il => _tapValues[il])]));
            }

            if (_kvExportValues is not null)
            {
                var exported = _kvExports.SelectMany(il => (Value[])[b.Reshape(_kvExportValues[il].K, 0, -1), b.Reshape(_kvExportValues[il].V, 0, -1)]);
                b.Output(SharedKvDraftHead.FeaturesOutput, b.Concat(1, [hidden, .. exported]));
            }
        }

        _tapValues = null;
        _mediaMask = null;
        _kvExportValues = null;
        _adapterSlots = null;

        return b.Build();
    }

    /// <summary>Whether the model is a masked-diffusion language model (bidirectional, <c>diffusion.shift_logits</c> declared).</summary>
    public bool IsDiffusion => !_hp.Causal && _src.Has($"{_src.Architecture}.diffusion.shift_logits");

    // Masked diffusion: the language model head over every row of one bidirectional pass (WellKnown.DenoiseGraph).
    private ComputeGraph BuildDenoise()
    {
        var b = new GraphBuilder(WellKnown.DenoiseGraph, _src.Weight);
        var inp = PackedSequences.Declare(b, tokenTypes: false);
        var x = PackedLayers(b, inp);
        using (b.Group("output"))
        {
            var hidden = OutputNorm(b, x);
            b.Output(WellKnown.LogitsOutput, Logits(b, hidden));
        }

        return b.Build();
    }

    // Embeddings and layers of the packed-sequence graphs (attention within each sequence, no KV cache).
    private Value PackedLayers(GraphBuilder b, PackedSequenceInputs inp)
    {
        Value x;
        Value? perLayer = null;
        using (b.Group("input"))
        {
            x = AddPositionEmbeddings(b, TokenEmbeddings(b, RouteAdapters(b, inp.Tokens, inp.RowSpans)), inp.Positions);
            if (_hp.PerLayerSize > 0)
            {
                perLayer = PerLayerInputs(b, x, inp.Tokens);
            }
        }

        var output = Layers(b, x, inp.Positions, FrequencyFactors(b), perLayer, inp.RowSpans);
        _adapterSlots = null;
        return output;
    }

    // Token-activated adapters: routes the rows (setting _adapterSlots for the projections, see Project) and returns the tokens to embed.
    // Without a router the tokens are returned as they are.
    private Value RouteAdapters(GraphBuilder b, Value tokens, Value? rowSpans)
    {
        if (!AdapterRouter.IsPresent(_src))
        {
            return tokens;
        }

        int adapters = _src.Arch("adapters.count", 0);
        var (embedded, slots) = AdapterRouter.Route(b, tokens, adapters, (q, k, v) =>
        {
            if (rowSpans is not null)
            {
                return PackedSequences.Attention(b, q, k, v, rowSpans, 1f, causal: true, name: "adapter_router_attn");
            }

            var (keyCache, valueCache) = _routerCache ?? throw new InvalidOperationException("decoder: adapter routing needs causal attention.");
            b.CacheWrite(b.State(keyCache), k);
            b.CacheWrite(b.State(valueCache), v);
            b.RuntimeScalar(WellKnown.PastScalar);
            return b.Emit1("CachedAttention", [q, b.State(keyCache), b.State(valueCache)],
                Attributes.From(("scale", 1f), ("offset", WellKnown.PastScalar)), "adapter_router_attn");
        });
        _adapterSlots = slots;
        return embedded;
    }

    // A linear projection plus, with adapter routing, its switched low-rank delta (SwitchedLora).
    private Value Project(GraphBuilder b, Value x, string weight, string? bias = null, string? name = null)
    {
        var projected = b.Linear(x, weight, bias, name);
        return _adapterSlots is null ? projected : SwitchedLora.AddDelta(b, projected, x, weight, _adapterSlots);
    }

    // The projection {prefix}{module} (Project), mixed per row with the layer's media weight set (TokenWeightSets) while the multimodal
    // graph is built.
    private Value ProjectSets(GraphBuilder b, Value x, string prefix, string module, string? name = null)
    {
        string text = prefix + module;
        if (_mediaMask is null || !TokenWeightSets.Has(b, prefix, module))
        {
            return Project(b, x, text + ".weight", text + ".bias", name);
        }

        string media = prefix + TokenWeightSets.MediaInfix + module;
        return TokenWeightSets.Mix(b, Project(b, x, text + ".weight", text + ".bias"), b.Linear(x, media + ".weight", media + ".bias"), _mediaMask, name);
    }

    private ComputeGraph BuildPacked(string name, bool classify)
    {
        var b = new GraphBuilder(name, _src.Weight);
        var inp = PackedSequences.Declare(b, tokenTypes: false);
        var x = PackedLayers(b, inp);
        using (b.Group("output"))
        {
            if (!classify)
            {
                var hidden = OutputNorm(b, x);
                PackedSequences.EmbeddingsOutput(b, hidden, inp.SequenceSpans, _pooling, _normalize);
            }
            else
            {
                // Score heads read the language model head at each sequence's last row (e.g. "yes" / "no" logits).
                var last = PackedSequences.Pool(b, x, inp.SequenceSpans, "last", "last_rows");
                var hidden = OutputNorm(b, last);
                b.Output(WellKnown.LogitsOutput, Logits(b, hidden));
            }
        }

        return b.Build();
    }

    // Token embedding lookup followed by the optional scale and weight-less norm some models apply to it.
    private Value TokenEmbeddings(GraphBuilder b, Value tokens)
    {
        _valueEmbeddings = b.TryWeight(ValueEmbeddingWeight) is { } valueTable ? b.Gather(valueTable, tokens, "inp_value_embd") : null;
        var x = b.Gather(b.Weight("token_embedding.weight"), tokens, "inp_embd");
        if (_hp.EmbeddingScale != 1f)
        {
            x = b.Scale(x, _hp.EmbeddingScale);
        }

        if (_hp.EmbeddingNorm)
        {
            x = b.RmsNorm(x, null, _hp.Eps);
        }

        if (b.HasWeight("embedding_norm.weight"))
        {
            x = _norm.Apply(b, x, "embedding_norm.weight", _hp.Eps);
        }

        return x;
    }

    // Learned absolute position embeddings (GPT-2, StarCoder): added to the token embeddings when the model has the table.
    private static Value AddPositionEmbeddings(GraphBuilder b, Value x, Value positions)
    {
        if (b.TryWeight("position_embedding.weight") is not { } table)
        {
            return x;
        }

        return b.Add(x, b.Gather(table, b.Reshape(positions, 0), "inp_pos_embd"));
    }

    private Value? FrequencyFactors(GraphBuilder b)
    {
        var freqFactors = b.TryWeight("rope.freq_factors");
        if (_longRope && b.TryWeight("rope.freq_factors_long") is { } longFactors)
        {
            freqFactors = longFactors;
        }

        return freqFactors;
    }

    private Value Logits(GraphBuilder b, Value hidden)
    {
        var head = b.TryWeight("lm_head.weight") ?? b.Weight("token_embedding.weight");
        var logits = b.MatMul(hidden, head, "logits");
        if (b.TryWeight("lm_head.bias") is { } headBias)
        {
            logits = b.Add(logits, headBias);
        }

        if (_hp.LogitScale != 1f)
        {
            logits = b.Scale(logits, _hp.LogitScale);
        }
        if (_hp.LogitSigmoidCap is { } cap)
        {
            // scale · sigmoid((logits + shift) / temperature) (Limite).
            logits = b.Scale(b.Sigmoid(b.Scale(logits, 1f / cap[2], cap[1] / cap[2])), cap[0], 0, "logits_capped");
        }

        if (_hp.FinalSoftCap > 0)
        {
            logits = b.Scale(b.Unary("Tanh", b.Scale(logits, 1f / _hp.FinalSoftCap)), _hp.FinalSoftCap, 0, "logits_capped");
        }

        return SuppressedLogits.Apply(b, logits, checked((int)head.Shape[0].Fixed), _hp.SuppressedTokenRanges);
    }

    // The transformer layers; rowSpans selects packed-sequence attention (no KV cache) instead of cached attention. With layer loops
    // (layer_loops) the physical layers run several times, each pass with KV caches of its own, optionally separated by the output norm.
    // deepstack: [T, H·K] blocks added to the hidden state after the first K layers (Qwen3-VL DeepStack).
    private Value Layers(GraphBuilder b, Value x, Value positions, Value? freqFactors, Value? perLayer, Value? rowSpans, Value? deepstack = null)
    {
        if (_hp.Cycles is { } cycles)
        {
            return CycleLayers(b, x, cycles, positions, freqFactors, rowSpans);
        }

        int deepstackLayers = deepstack is null ? 0 : (int)(deepstack.Shape[-1].Fixed / _hp.NEmbd);
        var packedKv = new Dictionary<int, (Value K, Value V)>();
        var embeddings = x;
        var history = _hp.DenseConnections is null ? null : new List<Value>();
        int streams = _hp.AltUpStreams;
        if (streams > 0)
        {
            x = AltUp.Expand(b, x, streams, _hp.NEmbd);
        }

        for (int logical = 0; logical < _hp.LogicalLayers; logical++)
        {
            int il = logical % _hp.NLayer;
            if (logical > 0 && il == 0 && _hp.LoopNorm)
            {
                x = _norm.Apply(b, x, "output_norm.weight", _hp.Eps, $"loop{logical / _hp.NLayer}_norm");
            }

            using var _ = b.Group($"layers.{logical}");
            string p = $"layers.{il}.";
            var slot = new LayerSlot(il, logical);
            _tapValues?.TryAdd(logical, x);
            if (streams > 0)
            {
                x = AltUpLayer(b, x, p, slot, positions, freqFactors, rowSpans, packedKv, embeddings, perLayer);
                continue;
            }

            Value? attentionInput = null;
            history?.Add(x);
            if (history is not null && _hp.DenseConnections!.TapsOf(il) is { } taps && DenseConnections.IsPresent(b, p))
            {
                // Dense connections (MUDD): the attention reads one mixture of earlier layer inputs, the residual stream another.
                var mixtures = DenseConnections.Mix(b, history, x, taps, p, _hp.Eps, mixtures: 2);
                attentionInput = mixtures[0];
                x = mixtures[1];
            }

            x = _hp.ParallelResidual
                ? ParallelLayer(b, x, p, slot, positions, freqFactors, rowSpans, packedKv, embeddings)
                : SequentialLayer(b, x, p, slot, positions, freqFactors, rowSpans, packedKv, embeddings, attentionInput);

            if (b.TryWeight(p + "embedding_skip.weight") is { } skipScale)
            {
                x = b.Add(x, b.Mul(embeddings, skipScale), $"{p}embedding_skip");
            }

            if (perLayer is not null && b.TryWeight(p + "per_layer.gate.weight") is not null)
            {
                x = b.Add(x, PerLayerDelta(b, x, perLayer, p, il), $"{p}per_layer_out");
            }

            if (b.TryWeight(p + "output_scale") is { } oscale)
            {
                x = b.Mul(x, oscale, $"{p}scaled");
            }

            if (logical < deepstackLayers)
            {
                x = b.Add(x, b.Slice(deepstack!, 1, (long)logical * _hp.NEmbd, _hp.NEmbd), $"{p}deepstack_out");
            }
        }

        return streams > 0 ? AltUp.Collapse(b, x, streams, _hp.NEmbd) : x;
    }

    // Hierarchical cycles (HRM-Text): the low and high stacks run alternately, every pass with KV caches of its own (see HierarchicalCycles).
    private Value CycleLayers(GraphBuilder b, Value x, HierarchicalCycleOptions cycles, Value positions, Value? freqFactors, Value? rowSpans)
    {
        var packedKv = new Dictionary<int, (Value K, Value V)>();
        var embeddings = x;
        return HierarchicalCycles.Build(b, x, cycles, _norm, _hp.Eps, (input, slotBase, layerBase) =>
        {
            var h = input;
            for (int i = 0; i < cycles.StackLayers; i++)
            {
                using var _ = b.Group($"layers.{slotBase + i}");
                string p = $"layers.{layerBase + i}.";
                var slot = new LayerSlot(layerBase + i, slotBase + i);
                h = _hp.ParallelResidual
                    ? ParallelLayer(b, h, p, slot, positions, freqFactors, rowSpans, packedKv, embeddings)
                    : SequentialLayer(b, h, p, slot, positions, freqFactors, rowSpans, packedKv, embeddings);
            }

            return h;
        });
    }

    // The final norm of the trunk. Hierarchical cycles end with the high stack's norm, so a model with cycles and no output norm weight has none.
    private Value OutputNorm(GraphBuilder b, Value x)
        => _hp.Cycles is not null && !b.HasWeight("output_norm.weight") ? x : _norm.Apply(b, x, "output_norm.weight", _hp.Eps, "result_norm");

    // One layer over AltUp streams [T, S·H] (Gemma 3n): the streams are mixed into predictions, the active prediction runs through the
    // sequential layer, its output corrects every prediction. The per-layer embedding contribution, gated by the active corrected stream
    // (times altup.correct_output_scale), is added to the other streams only.
    private Value AltUpLayer(GraphBuilder b, Value x, string p, LayerSlot slot, Value positions, Value? freqFactors, Value? rowSpans,
        Dictionary<int, (Value K, Value V)> packedKv, Value? embeddings, Value? perLayer)
    {
        int streams = _hp.AltUpStreams;
        int hidden = _hp.NEmbd;
        var predictions = AltUp.Predict(b, x, p, streams, hidden, _hp.Eps);
        var output = SequentialLayer(b, AltUp.Stream(b, predictions, 0, hidden), p, slot, positions, freqFactors, rowSpans, packedKv, embeddings);
        var corrected = AltUp.Correct(b, predictions, output, p, streams, hidden, _hp.Eps);
        if (perLayer is null || !b.HasWeight(p + "per_layer.gate.weight"))
        {
            return corrected;
        }

        var active = AltUp.Stream(b, corrected, 0, hidden);
        if (b.TryWeight(p + "altup.correct_output_scale") is { } scale)
        {
            active = b.Mul(active, scale);
        }

        return AltUp.AddToInactive(b, corrected, PerLayerDelta(b, active, perLayer, p, slot.Layer), streams, hidden, $"{p}per_layer_out");
    }

    // Residual topology "sequential": x += attention(norm(x)); x += mlp(norm(x)). A layer without attention weights skips the attention
    // (Deci); one whose attention is a single linear map (attention.o without q/k/v) adds o(norm(x)). A layer with moe_norm.weight adds a
    // mixture-of-experts branch that reads the layer input (moe(moe_norm(x)), Arctic) next to its dense MLP; a layer without MLP weights has none.
    // attentionInput: what the attention block reads instead of x (dense connections), x staying the residual stream.
    private Value SequentialLayer(GraphBuilder b, Value x, string p, LayerSlot slot, Value positions, Value? freqFactors, Value? rowSpans,
        Dictionary<int, (Value K, Value V)> packedKv, Value? embeddings, Value? attentionInput = null)
    {
        var residual = x;
        if (b.HasWeight(p + "attention.o.weight"))
        {
            var h = PreNorm(b, attentionInput ?? x, p + "input_norm.weight", $"{p}attn_norm");
            var attention = b.HasWeight(p + "attention.q.weight")
                ? Attention(b, h, p, slot, positions, freqFactors, rowSpans, packedKv, embeddings)
                : b.Linear(h, p + "attention.o.weight", p + "attention.o.bias", $"{p}attn_out");
            x = ResidualAdd(b, SubLayerOutput(b, attention, p + "attention.out_norm.weight"), residual, 2 * (slot.Layer + 1), p + "attention.", $"{p}attn_residual");
            if (b.HasWeight(p + "laurel.left.weight"))
            {
                x = Laurel(b, x, h, p);
            }
        }

        if (!HasFeedForward(b, p))
        {
            return x;
        }

        var ffnResidual = x;
        var normed = PreNorm(b, x, p + "post_attention_norm.weight", $"{p}post_attn_norm");
        var ffn = SubLayerOutput(b, FeedForward(b, normed, p, slot.Layer, residual), p + "mlp.out_norm.weight");
        var output = ResidualAdd(b, ffn, ffnResidual, 2 * (slot.Layer + 1) + 1, p + "mlp.", $"{p}l_out");
        if (b.HasWeight(p + "moe_norm.weight"))
        {
            var experts = MoeMlp(b, _norm.Apply(b, residual, p + "moe_norm.weight", _hp.Eps, $"{p}moe_norm"), p, slot.Layer, residual);
            output = b.Add(output, experts, $"{p}moe_residual");
        }

        return output;
    }

    // LAuReL (Gemma 3n): a low-rank residual of the attention input h next to the attention, (x + h + norm(right(left(h)))) / √2.
    private Value Laurel(GraphBuilder b, Value x, Value h, string p)
    {
        var lowRank = b.Linear(b.Linear(h, p + "laurel.left.weight"), p + "laurel.right.weight");
        var laurel = b.Add(h, _norm.Apply(b, lowRank, p + "laurel.norm.weight", _hp.Eps), $"{p}laurel");
        return b.Scale(b.Add(x, laurel), 1f / MathF.Sqrt(2f), 0, $"{p}attn_laurel");
    }

    // Residual topology "parallel": x += attention(norm(x)) + mlp(norm'(x)), both branches reading the layer input. The MLP uses
    // its own norm (post_attention_norm) when the layer has one, otherwise it shares the attention norm (Falcon-7B, Command-R, Phi-2).
    private Value ParallelLayer(GraphBuilder b, Value x, string p, LayerSlot slot, Value positions, Value? freqFactors, Value? rowSpans,
        Dictionary<int, (Value K, Value V)> packedKv, Value? embeddings)
    {
        var h = PreNorm(b, x, p + "input_norm.weight", $"{p}attn_norm");
        var attnOut = SubLayerOutput(b, Attention(b, h, p, slot, positions, freqFactors, rowSpans, packedKv, embeddings), p + "attention.out_norm.weight");
        var ffnInput = b.HasWeight(p + "post_attention_norm.weight") ? PreNorm(b, x, p + "post_attention_norm.weight", $"{p}post_attn_norm") : h;
        var ffn = SubLayerOutput(b, FeedForward(b, ffnInput, p, slot.Layer, x), p + "mlp.out_norm.weight");
        return b.Add(b.Add(attnOut, x, $"{p}attn_residual"), ffn, $"{p}l_out");
    }

    // The residual sum of a sequential layer's sub-layer output and its input. With residual.norm_preserving (Beam) the i-th sum (i = 2 for the
    // first attention, counting both sub-layers of every layer) is output / sqrt(i) + residual · sqrt((i − 1) / i), which keeps the expected
    // norm of the stream when the sub-layer outputs are normalized. A sub-layer with learned residual_alpha / residual_beta weights [H] (Muse Spark)
    // sums alpha · residual + beta · output instead (output alone without residual_beta, Limite).
    private Value ResidualAdd(GraphBuilder b, Value output, Value residual, int i, string block, string name)
    {
        if (b.TryWeight(block + "residual_alpha") is { } alpha)
        {
            var scaledOutput = b.TryWeight(block + "residual_beta") is { } beta ? b.Mul(output, beta) : output;
            return b.Add(b.Mul(residual, alpha), scaledOutput, name);
        }

        return _hp.NormPreservingResidual
            ? b.Add(b.Scale(output, (float)(1.0 / Math.Sqrt(i))), b.Scale(residual, (float)Math.Sqrt((i - 1.0) / i)), name)
            : b.Add(output, residual, name);
    }

    // Norm in front of a sub-layer; models with post-norms only (OLMo 2) declare pre_norm = false. A bidirectional layer without its norm weight
    // has none (ModernBERT's first layer reads the normed embeddings directly).
    private Value PreNorm(GraphBuilder b, Value x, string weightName, string name)
        => _hp.PreNorm && (_hp.Causal || b.HasWeight(weightName)) ? _norm.Apply(b, x, weightName, _hp.Eps, name) : x;

    // Optional norm of a sub-layer's output (sandwich / post norms), then the residual-branch scale (muP).
    private Value SubLayerOutput(GraphBuilder b, Value output, string outNormWeight)
    {
        if (b.HasWeight(outNormWeight))
        {
            // post_norm.weight_first (Muse Spark): the output is scaled by the weight and then normalized.
            output = _hp.PostNormWeightFirst
                ? b.RmsNorm(b.Mul(output, b.Weight(outNormWeight)), null, _hp.PostNormEps)
                : _norm.Apply(b, output, outNormWeight, _hp.PostNormEps);
        }

        return _hp.ResidualScale != 1f ? b.Scale(output, _hp.ResidualScale) : output;
    }

    // The routed experts when the layer has a router and no separate expert branch (moe_norm), the dense MLP otherwise.
    private Value FeedForward(GraphBuilder b, Value h, string p, int il, Value layerInput)
        => b.HasWeight(p + "mlp.router.weight") && !b.HasWeight(p + "moe_norm.weight") ? MoeMlp(b, h, p, il, layerInput) : Mlp(b, h, p, il);

    private static bool HasFeedForward(GraphBuilder b, string p) => b.HasWeight(p + "mlp.up.weight") || b.HasWeight(p + "mlp.router.weight");

    // Gated per-layer-embedding contribution that ends a layer (see the DecoderAdapter remarks), gated by x.
    private Value PerLayerDelta(GraphBuilder b, Value x, Value perLayer, string p, int il)
    {
        var gated = b.Mul(ActOf(b, b.Linear(x, p + "per_layer.gate.weight"), p),
                          b.Slice(perLayer, 1, (long)il * _hp.PerLayerSize, _hp.PerLayerSize));
        return b.RmsNorm(b.Linear(gated, p + "per_layer.projection.weight"), b.Weight(p + "per_layer.norm.weight"), _hp.Eps);
    }

    // Per-layer embeddings: [T, L*P] (see the DecoderAdapter remarks).
    private Value PerLayerInputs(GraphBuilder b, Value x, Value? tokens)
    {
        int P = _hp.PerLayerSize;
        int L = _hp.NLayer;
        var proj = b.Scale(b.MatMul(x, b.Weight("per_layer.model_projection.weight")), 1f / MathF.Sqrt(_hp.NEmbd));
        proj = b.Reshape(b.RmsNorm(b.Reshape(proj, 0, L, P), b.Weight("per_layer.projection_norm.weight"), _hp.Eps), 0, L * P);
        if (tokens is null || !b.HasWeight("per_layer.embedding.weight"))
        {
            return proj;                       // embedding input or projection-only PLE (EmbeddingGemma 2): no token identity
        }

        var tok = b.Scale(b.Gather(b.Weight("per_layer.embedding.weight"), tokens, "per_layer_tok"), MathF.Sqrt(P));
        return b.Scale(b.Add(proj, tok), 0.70710678f, 0, "per_layer_inputs");
    }
}
