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
    private Func<string, WeightInfo?>? _draft;
    private string _pooling = "last";
    private bool _normalize = true;

    /// <summary>Creates the builder and the KV-cache state definitions.</summary>
    /// <exception cref="InvalidDataException">A KV-sharing layer does not point to an earlier layer.</exception>
    public DecoderGraphBuilder(ModelSource source, AdapterOptions adapterOptions, DecoderHyperParameters hyperParameters)
    {
        _src = source;
        _options = adapterOptions;
        _hp = hyperParameters;
        _norm = new DecoderNorm(_hp.NormType);
        _activation = MlpActivations.Create(_hp);
        int origCtx = _src.Arch("rope.original_context_length", 0);
        int runtimeCtx = _options.ContextLength > 0 ? _options.ContextLength : _hp.ContextLength;
        _longRope = UseLongRope(origCtx, runtimeCtx);
        States = [];
        // A NextN draft head (one MTP layer after the trunk's, see MtpDraftHead.FirstNextnModule) when the layers are uniform: the
        // MTP layer is built like a trunk layer with index NLayer and has its own KV cache.
        _draft = _hp.UniformLayers && _hp.NormType == "rms" && _hp.PreNorm ? MtpDraftHead.FirstNextnModule(_src.Weight) : null;
        // Cache slots: one per logical layer (layer loops run the physical layers again with caches of their own), then the draft layer.
        int logical = _hp.LogicalLayers;
        int slots = logical + (_draft is null ? 0 : 1);
        _kCache = new StateDefinition?[slots];
        _vCache = new StateDefinition?[slots];
        for (int slot = 0; slot < slots; slot++)
        {
            int il = slot < logical ? slot % _hp.NLayer : _hp.NLayer;
            string g = slot < logical ? $"layers.{slot}" : $"mtp.layers.{slot - logical}";
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
        var graphs = new Dictionary<string, ComputeGraph> { [WellKnown.DecodeGraph] = BuildTrunk(WellKnown.DecodeGraph, embeddingsInput: false) };
        if (_draft is not null)
        {
            try
            {
                graphs[WellKnown.DraftGraph] = MtpDraftHead.Build(_draft, _hp.NEmbd, positionAxes: 1, _hp.Eps, layers: 1, (b, x, j, positions) =>
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

        if (_options.IncludeAuxiliaryGraphs)
        {
            graphs["decode_embeddings"] = BuildTrunk("decode_embeddings", embeddingsInput: true);
        }

        // Packed-sequence graphs (embedding models such as Qwen3-Embedding, yes/no rerankers such as Qwen3-Reranker): causal attention
        // within each sequence, no KV cache. Last-token pooling unless the model declares otherwise. Not built for models with
        // attention sinks (the packed attention has none).
        _pooling = PackedSequences.NormalizePooling(_options.EmbeddingPooling) ?? PackedSequences.NormalizePooling(_src.Get<string?>(CanonicalKeys.EmbeddingPooling, null)) ?? "last";
        _normalize = _options.EmbeddingNormalize ?? _src.Get(CanonicalKeys.EmbeddingNormalize, true);
        bool packable = !Enumerable.Range(0, _hp.NLayer).Any(il => _src.Weight($"layers.{il}.attention.sinks") is not null);
        if (_options.IncludeAuxiliaryGraphs && packable)
        {
            graphs[WellKnown.EmbedGraph] = BuildPacked(WellKnown.EmbedGraph, classify: false);
            graphs[WellKnown.ClassifyGraph] = BuildPacked(WellKnown.ClassifyGraph, classify: true);
        }

        return graphs;
    }

    private ComputeGraph BuildTrunk(string name, bool embeddingsInput)
    {
        var b = new GraphBuilder(name, _src.Weight);
        var T = Dim.Sym(WellKnown.TokenSymbol);
        Value x;
        Value positions;
        Value? perLayer = null;
        using (b.Group("input"))
        {
            Value? tokens = null;
            if (embeddingsInput)
            {
                x = b.Input("embeddings", InputRole.Embeddings, DataType.F32, new Shape(T, _hp.NEmbd));
                positions = b.Input("positions", InputRole.Positions, DataType.I32, new Shape(T, 1));
            }
            else
            {
                tokens = b.Input("tokens", InputRole.TokenIds, DataType.I32, new Shape(T));
                positions = b.Input("positions", InputRole.Positions, DataType.I32, new Shape(T, 1));
                x = TokenEmbeddings(b, tokens);
            }

            x = AddPositionEmbeddings(b, x, positions);
            if (_hp.PerLayerSize > 0)
            {
                perLayer = PerLayerInputs(b, x, tokens);
            }
        }

        var outRows = b.Input("output_rows", InputRole.OutputRows, DataType.I32, new Shape(Dim.Sym("R")));
        x = Layers(b, x, positions, FrequencyFactors(b), perLayer, rowSpans: null);

        using (b.Group("output"))
        {
            var hidden = _norm.Apply(b, x, "output_norm.weight", _hp.Eps, "result_norm");
            var rows = b.Gather(hidden, outRows, "selected_rows");
            b.Output("logits", Logits(b, rows));
            b.Output("hidden", hidden);
        }

        return b.Build();
    }

    private ComputeGraph BuildPacked(string name, bool classify)
    {
        var b = new GraphBuilder(name, _src.Weight);
        var inp = PackedSequences.Declare(b, tokenTypes: false);
        Value x;
        Value? perLayer = null;
        using (b.Group("input"))
        {
            x = AddPositionEmbeddings(b, TokenEmbeddings(b, inp.Tokens), inp.Positions);
            if (_hp.PerLayerSize > 0)
            {
                perLayer = PerLayerInputs(b, x, inp.Tokens);
            }
        }

        x = Layers(b, x, inp.Positions, FrequencyFactors(b), perLayer, inp.RowSpans);
        using (b.Group("output"))
        {
            if (!classify)
            {
                var hidden = _norm.Apply(b, x, "output_norm.weight", _hp.Eps, "result_norm");
                PackedSequences.EmbeddingsOutput(b, hidden, inp.SequenceSpans, _pooling, _normalize);
            }
            else
            {
                // Score heads read the language model head at each sequence's last row (e.g. "yes" / "no" logits).
                var last = PackedSequences.Pool(b, x, inp.SequenceSpans, "last", "last_rows");
                var hidden = _norm.Apply(b, last, "output_norm.weight", _hp.Eps, "result_norm");
                b.Output(WellKnown.LogitsOutput, Logits(b, hidden));
            }
        }

        return b.Build();
    }

    // Token embedding lookup followed by the optional scale and weight-less norm some models apply to it.
    private Value TokenEmbeddings(GraphBuilder b, Value tokens)
    {
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
        if (_hp.FinalSoftCap > 0)
        {
            logits = b.Scale(b.Unary("Tanh", b.Scale(logits, 1f / _hp.FinalSoftCap)), _hp.FinalSoftCap, 0, "logits_capped");
        }

        return logits;
    }

    // The transformer layers; rowSpans selects packed-sequence attention (no KV cache) instead of cached attention. With layer loops
    // (layer_loops) the physical layers run several times, each pass with KV caches of its own, optionally separated by the output norm.
    private Value Layers(GraphBuilder b, Value x, Value positions, Value? freqFactors, Value? perLayer, Value? rowSpans)
    {
        var packedKv = new Dictionary<int, (Value K, Value V)>();
        var embeddings = x;
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
            x = _hp.ParallelResidual
                ? ParallelLayer(b, x, p, slot, positions, freqFactors, rowSpans, packedKv, embeddings)
                : SequentialLayer(b, x, p, slot, positions, freqFactors, rowSpans, packedKv, embeddings);

            if (b.TryWeight(p + "embedding_skip.weight") is { } skipScale)
            {
                x = b.Add(x, b.Mul(embeddings, skipScale), $"{p}embedding_skip");
            }

            if (perLayer is not null && b.TryWeight(p + "per_layer.gate.weight") is not null)
            {
                x = PerLayerResidual(b, x, perLayer, p, il);
            }

            if (b.TryWeight(p + "output_scale") is { } oscale)
            {
                x = b.Mul(x, oscale, $"{p}scaled");
            }
        }

        return x;
    }

    // Residual topology "sequential": x += attention(norm(x)); x += mlp(norm(x)). A layer without attention weights skips the attention
    // (Deci); one whose attention is a single linear map (attention.o without q/k/v) adds o(norm(x)). A layer with moe_norm.weight adds a
    // mixture-of-experts branch that reads the layer input (moe(moe_norm(x)), Arctic) next to its dense MLP; a layer without MLP weights has none.
    private Value SequentialLayer(GraphBuilder b, Value x, string p, LayerSlot slot, Value positions, Value? freqFactors, Value? rowSpans,
        Dictionary<int, (Value K, Value V)> packedKv, Value? embeddings)
    {
        var residual = x;
        if (b.HasWeight(p + "attention.o.weight"))
        {
            var h = PreNorm(b, x, p + "input_norm.weight", $"{p}attn_norm");
            var attention = b.HasWeight(p + "attention.q.weight")
                ? Attention(b, h, p, slot, positions, freqFactors, rowSpans, packedKv, embeddings)
                : b.Linear(h, p + "attention.o.weight", p + "attention.o.bias", $"{p}attn_out");
            x = b.Add(SubLayerOutput(b, attention, p + "attention.out_norm.weight"), residual, $"{p}attn_residual");
        }

        if (!HasFeedForward(b, p))
        {
            return x;
        }

        var ffnResidual = x;
        var normed = PreNorm(b, x, p + "post_attention_norm.weight", $"{p}post_attn_norm");
        var ffn = SubLayerOutput(b, FeedForward(b, normed, p, slot.Layer, residual), p + "mlp.out_norm.weight");
        var output = b.Add(ffn, ffnResidual, $"{p}l_out");
        if (b.HasWeight(p + "moe_norm.weight"))
        {
            var experts = MoeMlp(b, _norm.Apply(b, residual, p + "moe_norm.weight", _hp.Eps, $"{p}moe_norm"), p, slot.Layer, residual);
            output = b.Add(output, experts, $"{p}moe_residual");
        }

        return output;
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

    // Norm in front of a sub-layer; models with post-norms only (OLMo 2) declare pre_norm = false.
    private Value PreNorm(GraphBuilder b, Value x, string weightName, string name)
        => _hp.PreNorm ? _norm.Apply(b, x, weightName, _hp.Eps, name) : x;

    // Optional norm of a sub-layer's output (sandwich / post norms), then the residual-branch scale (muP).
    private Value SubLayerOutput(GraphBuilder b, Value output, string outNormWeight)
    {
        if (b.HasWeight(outNormWeight))
        {
            output = _norm.Apply(b, output, outNormWeight, _hp.PostNormEps);
        }

        return _hp.ResidualScale != 1f ? b.Scale(output, _hp.ResidualScale) : output;
    }

    // The routed experts when the layer has a router and no separate expert branch (moe_norm), the dense MLP otherwise.
    private Value FeedForward(GraphBuilder b, Value h, string p, int il, Value layerInput)
        => b.HasWeight(p + "mlp.router.weight") && !b.HasWeight(p + "moe_norm.weight") ? MoeMlp(b, h, p, il, layerInput) : Mlp(b, h, p, il);

    private static bool HasFeedForward(GraphBuilder b, string p) => b.HasWeight(p + "mlp.up.weight") || b.HasWeight(p + "mlp.router.weight");

    // Gated per-layer-embedding contribution that ends a layer (see the DecoderAdapter remarks).
    private Value PerLayerResidual(GraphBuilder b, Value x, Value perLayer, string p, int il)
    {
        var gated = b.Mul(ActOf(b, b.Linear(x, p + "per_layer.gate.weight"), p),
                          b.Slice(perLayer, 1, (long)il * _hp.PerLayerSize, _hp.PerLayerSize));
        var delta = b.RmsNorm(b.Linear(gated, p + "per_layer.projection.weight"), b.Weight(p + "per_layer.norm.weight"), _hp.Eps);
        return b.Add(x, delta, $"{p}per_layer_out");
    }

    // Per-layer embeddings: [T, L*P] (see the DecoderAdapter remarks).
    private Value PerLayerInputs(GraphBuilder b, Value x, Value? tokens)
    {
        int P = _hp.PerLayerSize;
        int L = _hp.NLayer;
        var proj = b.Scale(b.MatMul(x, b.Weight("per_layer.model_projection.weight")), 1f / MathF.Sqrt(_hp.NEmbd));
        proj = b.Reshape(b.RmsNorm(b.Reshape(proj, 0, L, P), b.Weight("per_layer.projection_norm.weight"), _hp.Eps), 0, L * P);
        if (tokens is null)
        {
            return proj;                       // embedding input: no token identity available
        }

        var tok = b.Scale(b.Gather(b.Weight("per_layer.embedding.weight"), tokens, "per_layer_tok"), MathF.Sqrt(P));
        return b.Scale(b.Add(proj, tok), 0.70710678f, 0, "per_layer_inputs");
    }
}
