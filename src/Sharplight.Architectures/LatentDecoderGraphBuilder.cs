using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Emits the cached decode graph of the canonical <c>latent_decoder</c> architecture (vocabulary on <see cref="LatentDecoderAdapter"/>).
/// The feed-forward blocks live in <c>LatentDecoderGraphBuilder.Mlp.cs</c>.
/// </summary>
internal sealed partial class LatentDecoderGraphBuilder
{
    private readonly ModelSource _src;
    private readonly LatentDecoderHyperParameters _hp;
    private readonly LatentAttention _attention;
    private readonly ChannelDeltaAttention _delta;
    private readonly IMlpActivation _activation;
    private readonly StateDefinition?[] _caches;
    private readonly (StateDefinition Conv, StateDefinition State)?[] _deltaStates;
    private readonly bool _longRope;
    private readonly Func<string, WeightInfo?>? _draft;
    private readonly StateDefinition? _draftCache;
    private readonly SparseAttentionIndexer? _indexer;
    private readonly StateDefinition?[] _indexerCaches;
    private Value? _previousSelection;   // while a graph is built: the last indexer selection, which layers without their own indexer reuse
    private Func<Value, Value>? _indexerRotate;   // while a graph is built: the indexer's rotary embedding

    /// <summary>Creates the builder and the latent cache state of every layer.</summary>
    public LatentDecoderGraphBuilder(ModelSource source, AdapterOptions options, LatentDecoderHyperParameters hyperParameters)
    {
        _src = source;
        _hp = hyperParameters;
        _attention = new LatentAttention(_hp.Attention);
        _activation = MlpActivations.Create(_hp.Activation);
        int originalContext = _src.Arch("rope.original_context_length", 0);
        int runtimeContext = options.ContextLength > 0 ? options.ContextLength : _hp.ContextLength;
        _longRope = originalContext > 0 && runtimeContext > originalContext;
        // Delta-attention layers (Kimi-Linear, Ling 3) keep a convolution window and a recurrent state instead of a latent cache.
        _delta = new ChannelDeltaAttention(_src);
        _deltaStates = [.. Enumerable.Range(0, _hp.LayerCount).Select(il => ChannelDeltaAttention.IsPresent(_src.Weight, $"layers.{il}.")
            ? _delta.DefineStates($"layers.{il}") : ((StateDefinition, StateDefinition)?)null)];
        _caches = [.. Enumerable.Range(0, _hp.LayerCount).Select(il => _deltaStates[il] is null ? _attention.CreateCache($"layers.{il}", options.KvCacheType) : null)];
        // DSA indexer: only when the context exceeds its budget (below it every row keeps every visible position, the dense attention).
        if (_hp.Indexer is { } indexer && runtimeContext > indexer.TokenBudget)
        {
            _indexer = new SparseAttentionIndexer(indexer);
        }

        _indexerCaches = [.. Enumerable.Range(0, _hp.LayerCount).Select(il => _indexer is not null && _caches[il] is not null
            && SparseAttentionIndexer.IsPresent(_src.Weight, $"layers.{il}.")
                ? SparseAttentionIndexer.CreateKeyCache(_src.Weight, $"layers.{il}.", $"layers.{il}", options.KvCacheType)
                : null)];
        // The NextN draft head (DeepSeek-V3: one MTP layer after the trunk's, see MtpDraftHead.FirstNextnModule) with its own cache;
        // not with hyper-connections (the hidden state is several residual streams).
        _draft = _hp.HyperConnections is null ? MtpDraftHead.FirstNextnModule(_src.Weight) : null;
        if (_draft is not null)
        {
            _draftCache = _attention.CreateCache("mtp.layers.0", options.KvCacheType);
        }

        IncludeEmbeddingsGraph = options.IncludeAuxiliaryGraphs;
    }

    /// <summary>The persistent states (one latent cache per layer, and the draft layer's).</summary>
    public List<StateDefinition> States
    {
        get
        {
            var states = new List<StateDefinition>();
            for (int il = 0; il < _hp.LayerCount; il++)
            {
                if (_deltaStates[il] is { } delta)
                {
                    states.Add(delta.Conv);
                    states.Add(delta.State);
                }
                else
                {
                    states.Add(_caches[il]!);
                }

                if (_indexerCaches[il] is { } keys)
                {
                    states.Add(keys);
                }
            }

            if (_draftCache is not null)
            {
                states.Add(_draftCache);
            }

            return states;
        }
    }

    /// <summary>Whether the model has a NextN draft head (<see cref="WellKnown.DraftGraph"/>).</summary>
    public bool HasDraftHead => _draft is not null;

    private bool IncludeEmbeddingsGraph { get; }

    /// <summary>Builds the graphs: token decode, the draft head when the model has one and, with auxiliary graphs, decode from embeddings.</summary>
    public Dictionary<string, ComputeGraph> BuildGraphs()
    {
        var graphs = new Dictionary<string, ComputeGraph> { [WellKnown.DecodeGraph] = Build(WellKnown.DecodeGraph, embeddingsInput: false) };
        if (_draft is not null)
        {
            graphs[WellKnown.DraftGraph] = MtpDraftHead.Build(_draft, _hp.HiddenSize, positionAxes: 1, _hp.Eps, layers: 1,
                (b, x, j, positions) => Layer(b, x, $"mtp.layers.{j}.", -1, LatentAttention.HeadIndices(b, positions, _hp.Attention.Heads), Rotation(b, positions),
                    Temperature(b, positions)),
                chainNormalized: true);
        }

        if (IncludeEmbeddingsGraph)
        {
            graphs["decode_embeddings"] = Build("decode_embeddings", embeddingsInput: true);
        }

        return graphs;
    }

    private ComputeGraph Build(string name, bool embeddingsInput)
    {
        var b = new GraphBuilder(name, _src.Weight);
        var T = Dim.Sym(WellKnown.TokenSymbol);
        Value x;
        using (b.Group("input"))
        {
            if (embeddingsInput)
            {
                x = b.Input("embeddings", InputRole.Embeddings, DataType.F32, new Shape(T, _hp.HiddenSize));
            }
            else
            {
                var tokens = b.Input("tokens", InputRole.TokenIds, DataType.I32, new Shape(T));
                x = b.Gather(b.Weight("token_embedding.weight"), tokens, "inp_embd");
                if (_hp.EmbeddingScale != 1f)
                {
                    x = b.Scale(x, _hp.EmbeddingScale);
                }
            }
        }

        var positions = b.Input("positions", InputRole.Positions, DataType.I32, new Shape(T, 1));
        var outRows = b.Input("output_rows", InputRole.OutputRows, DataType.I32, new Shape(Dim.Sym("R")));
        var headIds = LatentAttention.HeadIndices(b, positions, _hp.Attention.Heads);
        var rotate = Rotation(b, positions);
        var temperature = Temperature(b, positions);
        _previousSelection = null;
        _indexerRotate = _indexer is null ? null : IndexerRotation(b, positions);
        if (_hp.HyperConnections is { } hc)
        {
            x = StreamHyperConnections.Expand(b, x, hc.Streams);
        }

        for (int il = 0; il < _hp.LayerCount; il++)
        {
            using var _ = b.Group($"layers.{il}");
            x = Layer(b, x, $"layers.{il}.", il, headIds, rotate, temperature);
        }

        using (b.Group("output"))
        {
            if (_hp.HyperConnections is { } streams)
            {
                x = StreamHyperConnections.Head(b, x, "output_hc.", streams, _hp.HiddenSize);
            }

            var hidden = b.RmsNorm(x, b.Weight("output_norm.weight"), _hp.Eps, "result_norm");
            var rows = b.Gather(hidden, outRows, "selected_rows");
            if (_hp.LogitScale != 1f)
            {
                rows = b.Scale(rows, _hp.LogitScale);
            }

            var head = b.TryWeight("lm_head.weight") ?? b.Weight("token_embedding.weight");
            b.Output("logits", b.MatMul(rows, head, "logits"));
            b.Output("hidden", hidden);
        }

        return b.Build();
    }

    // One decoder layer: weights below p; il is the trunk layer (its latent cache or delta-attention states), -1 the draft layer.
    private Value Layer(GraphBuilder b, Value x, string p, int il, Value headIds, Func<Value, Value> rotate, Value? temperature)
    {
        if (_hp.HyperConnections is { } hc)
        {
            return HyperConnectedLayer(b, x, p, il, headIds, rotate, temperature, hc);
        }

        var h = b.RmsNorm(x, b.Weight(p + "input_norm.weight"), _hp.Eps, $"{p}attn_norm");
        var attention = Mixer(b, h, p, il, headIds, rotate, temperature);
        x = b.Add(ResidualBranch(b, attention), x, $"{p}attn_residual");

        h = b.RmsNorm(x, b.Weight(p + "post_attention_norm.weight"), _hp.Eps, $"{p}post_attn_norm");
        var ffn = MoeBlock.IsPresent(b, p + "mlp.") ? Experts(b, h, p) : Mlp(b, h, p, "mlp.", $"{p}ffn_out");
        return b.Add(ResidualBranch(b, ffn), x, $"{p}l_out");
    }

    // The token mixer of a layer: delta attention for a trunk layer that has it, latent attention otherwise (sparse with an indexer; the draft
    // layer attends densely: its proposals are verified by the trunk).
    private Value Mixer(GraphBuilder b, Value h, string p, int il, Value headIds, Func<Value, Value> rotate, Value? temperature)
    {
        if (il >= 0 && _deltaStates[il] is { } delta)
        {
            return _delta.Build(b, h, p, delta.Conv, delta.State);
        }

        return _attention.Build(b, h, p, il < 0 ? _draftCache! : _caches[il]!, headIds, rotate, temperature, il < 0 ? null : Selection(b, h, p, il));
    }

    /// <summary>
    /// The selection of a trunk layer's sparse attention: its own indexer's, or the previous indexed layer's when it has none (Hunyuan v4's
    /// shared layers); null without an indexer.
    /// </summary>
    private Func<Value, Value>? Selection(GraphBuilder b, Value h, string p, int il)
    {
        if (_indexer is null)
        {
            return null;
        }

        if (_indexerCaches[il] is { } keys)
        {
            return queryInput => _previousSelection = _indexer.Select(b, h, queryInput, p, keys, _hp.Attention.RopeDims, _indexerRotate!);
        }

        return _ => _previousSelection ?? throw new InvalidDataException($"latent_decoder: layer {il} reuses an indexer selection, but no earlier layer has an indexer.");
    }

    // The layer on hyper-connected residual streams [T, N·H] (StreamHyperConnections modules attention_hc / mlp_hc around the sub-layers).
    private Value HyperConnectedLayer(GraphBuilder b, Value x, string p, int il, Value headIds, Func<Value, Value> rotate, Value? temperature,
        StreamHyperConnectionOptions hc)
    {
        var (input, post, mixing) = StreamHyperConnections.Collapse(b, x, p + "attention_hc.", hc, _hp.HiddenSize);
        var h = b.RmsNorm(input, b.Weight(p + "input_norm.weight"), _hp.Eps, $"{p}attn_norm");
        var attention = Mixer(b, h, p, il, headIds, rotate, temperature);
        x = StreamHyperConnections.Merge(b, x, ResidualBranch(b, attention), post, mixing, hc.Streams, _hp.HiddenSize, $"{p}attn_residual");

        (input, post, mixing) = StreamHyperConnections.Collapse(b, x, p + "mlp_hc.", hc, _hp.HiddenSize);
        h = b.RmsNorm(input, b.Weight(p + "post_attention_norm.weight"), _hp.Eps, $"{p}post_attn_norm");
        var ffn = MoeBlock.IsPresent(b, p + "mlp.") ? Experts(b, h, p) : Mlp(b, h, p, "mlp.", $"{p}ffn_out");
        return StreamHyperConnections.Merge(b, x, ResidualBranch(b, ffn), post, mixing, hc.Streams, _hp.HiddenSize, $"{p}l_out");
    }

    /// <summary>The query temperature of every row (Mistral 4 / Mistral Large 3), or null when the model has none.</summary>
    private Value? Temperature(GraphBuilder b, Value positions)
        => _hp.TemperatureScale == 0f ? null : AttentionTemperature.Build(b, positions, _hp.TemperatureScale, _hp.TemperatureLength);

    private Value ResidualBranch(GraphBuilder b, Value branch) => _hp.ResidualScale == 1f ? branch : b.Scale(branch, _hp.ResidualScale);

    /// <summary>The rotary embedding of the rope parts ([T, heads, rope_dim]), with the frequency factors the context selects.</summary>
    private Func<Value, Value> Rotation(GraphBuilder b, Value positions)
    {
        if (_hp.RopeStyle == "none")
        {
            return t => t;   // no positional rotation (Kimi-Linear's latent attention)
        }

        return RopeOf(b, positions, _hp.RopeStyle, 1f);
    }

    /// <summary>The indexer's rotary embedding: the attention's frequencies and magnitude with the indexer's pairing style.</summary>
    private Func<Value, Value> IndexerRotation(GraphBuilder b, Value positions)
        => RopeOf(b, positions, _indexer!.Options.RopeStyle, _hp.Attention.RopeMagnitude);

    private Func<Value, Value> RopeOf(GraphBuilder b, Value positions, string style, float magnitude)
    {
        var factors = b.TryWeight("rope.freq_factors");
        if (_longRope && b.TryWeight("rope.freq_factors_long") is { } longFactors)
        {
            factors = longFactors;
        }

        var attributes = Attributes.From(("dims", _hp.Attention.RopeDims), ("base", _hp.RopeBase), ("style", style));
        if (magnitude != 1f)
        {
            attributes = attributes.With("attn_factor", magnitude);
        }

        return t => b.Emit1("Rope", factors is null ? [t, positions] : [t, positions, factors], attributes);
    }
}
