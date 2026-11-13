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
    private readonly StateDefinition[] _kCache;
    private readonly StateDefinition[] _vCache;
    private string _pooling = "last";
    private bool _normalize = true;

    /// <summary>Creates the builder and the KV-cache state definitions.</summary>
    /// <exception cref="InvalidDataException">A KV-sharing layer does not point to an earlier layer.</exception>
    public DecoderGraphBuilder(ModelSource source, AdapterOptions adapterOptions, DecoderHyperParameters hyperParameters)
    {
        _src = source;
        _options = adapterOptions;
        _hp = hyperParameters;
        int origCtx = _src.Arch("rope.original_context_length", 0);
        int runtimeCtx = _options.ContextLength > 0 ? _options.ContextLength : _hp.ContextLength;
        _longRope = UseLongRope(origCtx, runtimeCtx);

        States = [];
        _kCache = new StateDefinition[_hp.NLayer];
        _vCache = new StateDefinition[_hp.NLayer];
        for (int il = 0; il < _hp.NLayer; il++)
        {
            string g = $"layers.{il}";
            int srcLayer = _hp.KvSourceOf(il);
            if (srcLayer != il)
            {
                if (srcLayer > il || srcLayer < 0)
                {
                    throw new InvalidDataException("decoder: attention.kv_source_layers must point to an earlier layer.");
                }

                _kCache[il] = _kCache[srcLayer];
                _vCache[il] = _vCache[srcLayer];
                continue;
            }
            var shape = new Shape(Dim.Sym(WellKnown.ContextSymbol), _hp.KvHeadsOf(il), _hp.HeadDimOf(il));
            States.Add(_kCache[il] = new StateDefinition { Name = $"{g}.k_cache", Type = _options.KvCacheType, Shape = shape, Kind = StateKind.KeyCache, Group = g });
            States.Add(_vCache[il] = new StateDefinition { Name = $"{g}.v_cache", Type = _options.KvCacheType, Shape = shape, Kind = StateKind.ValueCache, Group = g });
        }
    }

    /// <summary>The persistent states (KV caches) the graphs use.</summary>
    public List<StateDefinition> States { get; }

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
        Value? perLayer = null;
        using (b.Group("input"))
        {
            Value? tokens = null;
            if (embeddingsInput)
            {
                x = b.Input("embeddings", InputRole.Embeddings, DataType.F32, new Shape(T, _hp.NEmbd));
            }
            else
            {
                tokens = b.Input("tokens", InputRole.TokenIds, DataType.I32, new Shape(T));
                x = TokenEmbeddings(b, tokens);
            }

            if (_hp.PerLayerSize > 0)
            {
                perLayer = PerLayerInputs(b, x, tokens);
            }
        }

        var positions = b.Input("positions", InputRole.Positions, DataType.I32, new Shape(T, 1));
        var outRows = b.Input("output_rows", InputRole.OutputRows, DataType.I32, new Shape(Dim.Sym("R")));
        x = Layers(b, x, positions, FrequencyFactors(b), perLayer, rowSpans: null);

        using (b.Group("output"))
        {
            var hidden = b.RmsNorm(x, b.Weight("output_norm.weight"), _hp.Eps, "result_norm");
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
            x = TokenEmbeddings(b, inp.Tokens);
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
                var hidden = b.RmsNorm(x, b.Weight("output_norm.weight"), _hp.Eps, "result_norm");
                PackedSequences.EmbeddingsOutput(b, hidden, inp.SequenceSpans, _pooling, _normalize);
            }
            else
            {
                // Score heads read the language model head at each sequence's last row (e.g. "yes" / "no" logits).
                var last = PackedSequences.Pool(b, x, inp.SequenceSpans, "last", "last_rows");
                var hidden = b.RmsNorm(last, b.Weight("output_norm.weight"), _hp.Eps, "result_norm");
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

        return x;
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

    // The transformer layers; rowSpans selects packed-sequence attention (no KV cache) instead of cached attention.
    private Value Layers(GraphBuilder b, Value x, Value positions, Value? freqFactors, Value? perLayer, Value? rowSpans)
    {
        var packedKv = new Dictionary<int, (Value K, Value V)>();
        for (int il = 0; il < _hp.NLayer; il++)
        {
            using var _ = b.Group($"layers.{il}");
            string p = $"layers.{il}.";
            var residual = x;
            var h = b.RmsNorm(x, b.Weight(p + "input_norm.weight"), _hp.Eps, $"{p}attn_norm");
            var attnOut = Attention(b, h, p, il, positions, freqFactors, rowSpans, packedKv);
            if (b.TryWeight(p + "attention.out_norm.weight") is { } on1)
            {
                attnOut = b.RmsNorm(attnOut, on1, _hp.PostNormEps);
            }

            x = b.Add(attnOut, residual, $"{p}attn_residual");

            var ffnResidual = x;
            h = b.RmsNorm(x, b.Weight(p + "post_attention_norm.weight"), _hp.Eps, $"{p}post_attn_norm");
            var ffn = b.HasWeight(p + "mlp.router.weight") ? MoeMlp(b, h, p) : Mlp(b, h, p);
            if (b.TryWeight(p + "mlp.out_norm.weight") is { } on2)
            {
                ffn = b.RmsNorm(ffn, on2, _hp.PostNormEps);
            }

            x = b.Add(ffn, ffnResidual, $"{p}l_out");

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

    // Gated per-layer-embedding contribution that ends a layer (see the DecoderAdapter remarks).
    private Value PerLayerResidual(GraphBuilder b, Value x, Value perLayer, string p, int il)
    {
        var gated = b.Mul(ActOf(b, b.Linear(x, p + "per_layer.gate.weight")),
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
