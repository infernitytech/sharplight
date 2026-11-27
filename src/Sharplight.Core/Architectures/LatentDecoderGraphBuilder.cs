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
    private readonly LatentAttention[] _attentions;   // per trunk layer: NoPE layers have no rotary dimensions (the whole head is non-rotary)
    private readonly bool _ssmax;
    private readonly ChannelDeltaAttention _delta;
    private readonly IMlpActivation _activation;
    private readonly StateDefinition?[] _caches;
    private readonly (StateDefinition Conv, StateDefinition State)?[] _deltaStates;
    private readonly bool _longRope;
    private readonly Func<string, WeightInfo?>? _draft;
    private readonly StateDefinition? _draftCache;
    private readonly SparseAttentionIndexer? _indexer;
    private readonly AttentionResiduals? _residuals;   // Kimi K3: block attention residuals instead of the running residual sum
    private readonly StateDefinition?[] _indexerCaches;
    private Value? _previousSelection;   // while a graph is built: the last indexer selection, which layers without their own indexer reuse
    private Func<Value, Value>? _indexerRotate;   // while a graph is built: the indexer's rotary embedding
    private Value? _logPositions;   // while a graph is built: ln(pos + 1) of every row for the SSMax factors
    private Value? _slidingHeadIds;   // while a graph is built: the head indices of the sliding-window layers
    private Func<Value, Value>? _slidingRotate;   // while a graph is built: the rotary embedding of the sliding-window layers
    private readonly CompressedKvAttention? _compressedKv;   // DeepSeek-V4: windowed attention with compressed entries instead of the latent cache
    private readonly CompressedKvLayerStates?[] _compressedStates;
    private Value? _positions;   // while a graph is built: the row positions
    private Value? _tokens;   // while a graph is built: the token ids (null for the embeddings graph), for hash routing
    private Value? _groupIds;   // while a graph is built: the output group indices of compressed-KV attention
    private readonly EngramOptions? _engram;   // DeepSeek-V4.1: hashed n-gram memory read by some layers
    private readonly StateDefinition? _engramWindow;
    private Value? _engramIds;   // while a graph is built: the n-gram token window (null for the embeddings graph)
    private Value? _laggedPre;   // while a graph is built: the collapse weights of the last module (lagged hyper-connections)
    private CompressedKvShared? _compressedShared;   // while a graph is built: caches and selections shared across layers
    private readonly float _steeringScale;   // Raon-1: the reasoning-effort factor of the layers' steering vectors (mlp.steering)

    /// <summary>Creates the builder and the latent cache state of every layer.</summary>
    public LatentDecoderGraphBuilder(ModelSource source, AdapterOptions options, LatentDecoderHyperParameters hyperParameters)
    {
        _src = source;
        _hp = hyperParameters;
        _attention = new LatentAttention(_hp.Attention);
        var nope = new LatentAttention(_hp.Attention with { NopeDims = _hp.Attention.NopeDims + _hp.Attention.RopeDims, RopeDims = 0 });
        var sliding = new LatentAttention(_hp.SlidingAttention);
        _attentions = [.. Enumerable.Range(0, _hp.LayerCount).Select(il => _hp.IsSliding(il) ? sliding : _hp.HasRope(il) ? _attention : nope)];
        _ssmax = Enumerable.Range(0, _hp.LayerCount).Any(il => _src.Weight($"layers.{il}.attention.ssmax.weight") is not null);
        _activation = MlpActivations.Create(_hp.Activation, _src);
        int originalContext = _src.Arch("rope.original_context_length", 0);
        int runtimeContext = options.ContextLength > 0 ? options.ContextLength : _hp.ContextLength;
        _longRope = originalContext > 0 && runtimeContext > originalContext;
        // Delta-attention layers (Kimi-Linear, Ling 3) keep a convolution window and a recurrent state instead of a latent cache.
        _delta = new ChannelDeltaAttention(_src);
        _residuals = AttentionResiduals.Read(_src, _hp.Eps);
        _steeringScale = _src.Arch("mlp.steering_scale", 0f);
        _deltaStates = [.. Enumerable.Range(0, _hp.LayerCount).Select(il => ChannelDeltaAttention.IsPresent(_src.Weight, $"layers.{il}.")
            ? _delta.DefineStates($"layers.{il}") : ((StateDefinition, StateDefinition)?)null)];
        _caches = new StateDefinition?[_hp.LayerCount];
        _engram = _hp.HyperConnections is null ? null : EngramOptions.Read(_src);
        _engramWindow = _engram is null ? null : EngramMemory.WindowState(_engram);
        _compressedStates = new CompressedKvLayerStates?[_hp.LayerCount];
        if (_hp.CompressedKv is { } compressedKv)
        {
            _compressedKv = new CompressedKvAttention(compressedKv, _hp.Attention.Heads, _hp.Attention.RopeDims, _hp.Eps);
            for (int il = 0; il < _hp.LayerCount; il++)
            {
                _compressedStates[il] = _compressedKv.CreateStates(_src.Weight, $"layers.{il}.", $"layers.{il}", il, options.KvCacheType, _compressedStates);
            }
        }

        for (int il = 0; il < _hp.LayerCount && _compressedKv is null; il++)
        {
            // Layers without their own key-value projection (Mistral Large 4) attend the cache of their source layer.
            int kvSource = _hp.KvSourceOf(il);
            _caches[il] = _deltaStates[il] is not null ? null
                : kvSource != il ? _caches[kvSource] : _attentions[il].CreateCache($"layers.{il}", options.KvCacheType);
        }

        // DSA indexer: only when the context exceeds its budget (below it every row keeps every visible position, the dense attention).
        if (_hp.Indexer is { } indexer && runtimeContext > indexer.TokenBudget && _compressedKv is null)
        {
            _indexer = new SparseAttentionIndexer(indexer);
        }

        _indexerCaches = [.. Enumerable.Range(0, _hp.LayerCount).Select(il => _indexer is not null && _caches[il] is not null && _compressedKv is null
            && SparseAttentionIndexer.IsPresent(_src.Weight, $"layers.{il}.")
                ? _indexer.CreateKeyCache(_src.Weight, $"layers.{il}.", $"layers.{il}", options.KvCacheType)
                : null)];
        // The NextN draft head (DeepSeek-V3: one MTP layer after the trunk's, see MtpDraftHead.FirstNextnModule) with its own cache;
        // not with hyper-connections (the hidden state is several residual streams).
        // Not with two attention geometries either (dots.3's NextN layer has the sliding one, which llama.cpp does not run).
        // Not with attention residuals either (the draft layer's residual would need the trunk's banked checkpoints).
        _draft = _hp.HyperConnections is null && _residuals is null && _hp.SlidingLayers.Length == 0 ? MtpDraftHead.FirstNextnModule(_src.Weight) : null;
        if (_draft is not null)
        {
            _draftCache = _attention.CreateCache("mtp.layers.0", options.KvCacheType);
        }

        // Hash-routed experts need the token ids, which the embeddings graph does not have.
        IncludeEmbeddingsGraph = options.IncludeAuxiliaryGraphs
            && !Enumerable.Range(0, _hp.LayerCount).Any(il => _src.Weight($"layers.{il}.mlp.router.token_experts") is not null);
    }

    /// <summary>The persistent states (one latent cache per layer, and the draft layer's).</summary>
    public List<StateDefinition> States
    {
        get
        {
            var states = new List<StateDefinition>();
            for (int il = 0; il < _hp.LayerCount; il++)
            {
                if (_compressedStates[il] is { } compressed)
                {
                    states.AddRange(compressed.All());
                    continue;
                }

                if (_deltaStates[il] is { } delta)
                {
                    states.Add(delta.Conv);
                    states.Add(delta.State);
                }
                else if (_hp.KvSourceOf(il) == il)
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

            if (_engramWindow is not null)
            {
                states.Add(_engramWindow);
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
        _tokens = null;
        if (_draft is not null)
        {
            graphs[WellKnown.DraftGraph] = MtpDraftHead.Build(_draft, _hp.HiddenSize, positionAxes: 1, _hp.Eps, layers: 1,
                (b, x, j, positions) =>
                {
                    _logPositions = _ssmax ? ScalableSoftmax.LogPositions(b, positions) : null;
                    return Layer(b, x, $"mtp.layers.{j}.", -1, LatentAttention.HeadIndices(b, positions, _hp.Attention.Heads), Rotation(b, positions),
                        Temperature(b, positions));
                },
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
                _tokens = tokens;
                x = b.Gather(b.Weight("token_embedding.weight"), tokens, "inp_embd");
                if (_hp.EmbeddingScale != 1f)
                {
                    x = b.Scale(x, _hp.EmbeddingScale);
                }
            }
        }

        var positions = b.Input("positions", InputRole.Positions, DataType.I32, new Shape(T, 1));
        var outRows = b.Input("output_rows", InputRole.OutputRows, DataType.I32, new Shape(Dim.Sym("R")));
        _positions = positions;
        _groupIds = _compressedKv is null ? null : LatentAttention.HeadIndices(b, positions, _compressedKv.Options.OutputGroups, "_groups");
        var headIds = LatentAttention.HeadIndices(b, positions, _hp.Attention.Heads);
        var rotate = Rotation(b, positions);
        var temperature = Temperature(b, positions);
        _previousSelection = null;
        _indexerRotate = _indexer is null ? null : IndexerRotation(b, positions);
        _residuals?.Reset();
        _logPositions = _ssmax ? ScalableSoftmax.LogPositions(b, positions) : null;
        _slidingHeadIds = _hp.SlidingAttention.Heads == _hp.Attention.Heads ? headIds : LatentAttention.HeadIndices(b, positions, _hp.SlidingAttention.Heads, "_sliding");
        _slidingRotate = _hp.SlidingLayers.Length == 0 ? null : RopeOf(b, positions, _hp.RopeStyle, 1f, _hp.SlidingRopeBase, "rope.freq_factors_sliding");
        _laggedPre = null;
        _compressedShared = _compressedKv is null ? null : new CompressedKvShared();
        _engramIds = _engram is not null && !embeddingsInput ? EngramMemory.Window(b, _tokens!, _engram, b.State(_engramWindow!)) : null;
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
            if (_residuals is not null)
            {
                x = _residuals.Mix(b, x, "output_res.score.weight", _hp.HiddenSize, "output_res");
            }

            if (_hp.HyperConnections is { } streams)
            {
                // A learned head collapses the streams with its own weights; without one (GLM-5.3-Flash) they are averaged.
                // Lagged collapse weights (DeepSeek-V4.1): the last module's collapse the final streams.
                x = streams.LaggedCollapse ? StreamHyperConnections.Apply(b, x, _laggedPre!, streams, _hp.HiddenSize, "output_hc.output")
                    : b.HasWeight("output_hc.fn.weight")
                    ? StreamHyperConnections.Head(b, x, "output_hc.", streams, _hp.HiddenSize)
                    : StreamHyperConnections.Mean(b, x, streams.Streams, _hp.HiddenSize);
            }

            var hidden = b.RmsNorm(x, b.Weight("output_norm.weight"), _hp.Eps, "result_norm");
            var rows = b.Gather(hidden, outRows, "selected_rows");
            if (_hp.LogitScale != 1f)
            {
                rows = b.Scale(rows, _hp.LogitScale);
            }

            var head = b.TryWeight("lm_head.weight") ?? b.Weight("token_embedding.weight");
            var logits = b.MatMul(rows, head, "logits");
            if (_hp.FinalSoftCap > 0f)
            {
                logits = b.Scale(b.Unary("Tanh", b.Scale(logits, 1f / _hp.FinalSoftCap)), _hp.FinalSoftCap, 0f, "logits_capped");
            }

            b.Output("logits", logits);
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

        if (_residuals is not null && il >= 0)
        {
            return ResidualAttentionLayer(b, x, p, il, headIds, rotate, temperature, _residuals);
        }

        var h = b.RmsNorm(x, b.Weight(p + "input_norm.weight"), _hp.Eps, $"{p}attn_norm");
        var attention = Mixer(b, h, p, il, headIds, rotate, temperature);
        x = b.Add(ResidualBranch(b, attention), x, $"{p}attn_residual");

        h = b.RmsNorm(x, b.Weight(p + "post_attention_norm.weight"), _hp.Eps, $"{p}post_attn_norm");
        var ffn = FeedForward(b, h, p);
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

        if (il >= 0 && _compressedStates[il] is { } compressed)
        {
            // Sliding-only layers rotate with the main rope, layers with compressed entries with the compressed one.
            bool main = _compressedKv!.Options.Ratios[il] == 0;
            var rope = RopeAt(b, main ? _hp.RopeBase : _compressedKv.Options.CompressedRopeBase, main ? "rope.freq_factors" : "rope.freq_factors_compressed");
            return _compressedKv.Build(b, h, p, il, compressed, _positions!, _groupIds!, rope, _compressedShared!);
        }

        if (il >= 0 && _hp.IsSliding(il))
        {
            // Sliding-window layers (dots.3): their own geometry and rope base, dense attention over the window.
            return _attentions[il].Build(b, h, p, _caches[il]!, _slidingHeadIds!, _slidingRotate!, temperature, null, _logPositions);
        }

        var attention = il < 0 ? _attention : _attentions[il];
        return attention.Build(b, h, p, il < 0 ? _draftCache! : _caches[il]!, headIds, rotate, temperature, il < 0 ? null : Selection(b, h, p, il), _logPositions);
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
            // An indexer without rotary embedding (Kimi K3.1) scores the whole rows as they are.
            int ropeDims = _indexer.Options.RopeStyle == "none" ? 0 : _attentions[il].Parameters.RopeDims;
            return queryInput => _previousSelection = _indexer.Select(b, h, queryInput, p, keys, ropeDims, _indexerRotate!, _logPositions);
        }

        return _ => _previousSelection ?? throw new InvalidDataException($"latent_decoder: layer {il} reuses an indexer selection, but no earlier layer has an indexer.");
    }

    // The layer on hyper-connected residual streams [T, N·H] (StreamHyperConnections modules attention_hc / mlp_hc around the sub-layers).
    private Value HyperConnectedLayer(GraphBuilder b, Value x, string p, int il, Value headIds, Func<Value, Value> rotate, Value? temperature,
        StreamHyperConnectionOptions hc)
    {
        if (il >= 0 && _engramIds is not null && _engram!.ModuleOf(il) is int module and >= 0)
        {
            x = EngramMemory.Build(b, _engramIds, x, p, module, _engram, hc.Streams, _hp.HiddenSize, _hp.Eps);
        }

        if (hc.LaggedCollapse)
        {
            return LaggedHyperConnectedLayer(b, x, p, il, headIds, rotate, temperature, hc);
        }

        var (input, post, mixing) = StreamHyperConnections.Collapse(b, x, p + "attention_hc.", hc, _hp.HiddenSize);
        var h = b.RmsNorm(input, b.Weight(p + "input_norm.weight"), _hp.Eps, $"{p}attn_norm");
        var attention = Mixer(b, h, p, il, headIds, rotate, temperature);
        x = StreamHyperConnections.Merge(b, x, ResidualBranch(b, attention), post, mixing, hc, _hp.HiddenSize, $"{p}attn_residual");

        (input, post, mixing) = StreamHyperConnections.Collapse(b, x, p + "mlp_hc.", hc, _hp.HiddenSize);
        h = b.RmsNorm(input, b.Weight(p + "post_attention_norm.weight"), _hp.Eps, $"{p}post_attn_norm");
        var ffn = FeedForward(b, h, p);
        return StreamHyperConnections.Merge(b, x, ResidualBranch(b, ffn), post, mixing, hc, _hp.HiddenSize, $"{p}l_out");
    }

    // The layer on single-pass hyper-connections (DeepSeek-V4.1): each module's collapse weights, computed from its block's input, collapse the
    // next block's input (the attention's from the previous layer's FFN module, the first layer's attention reads stream 0).
    private Value LaggedHyperConnectedLayer(GraphBuilder b, Value x, string p, int il, Value headIds, Func<Value, Value> rotate, Value? temperature,
        StreamHyperConnectionOptions hc)
    {
        int hidden = _hp.HiddenSize;
        var (attentionPre, post, mixing) = StreamHyperConnections.Weigh(b, x, p + "attention_hc.", hc);
        var input = _laggedPre is null ? StreamHyperConnections.Stream(b, x, 0, hidden)
            : StreamHyperConnections.Apply(b, x, _laggedPre, hc, hidden, $"{p}attention_hc.input");
        var h = b.RmsNorm(input, b.Weight(p + "input_norm.weight"), _hp.Eps, $"{p}attn_norm");
        var attention = Mixer(b, h, p, il, headIds, rotate, temperature);
        x = StreamHyperConnections.Merge(b, x, ResidualBranch(b, attention), post, mixing, hc, hidden, $"{p}attn_residual");

        (_laggedPre, post, mixing) = StreamHyperConnections.Weigh(b, x, p + "mlp_hc.", hc);
        h = b.RmsNorm(StreamHyperConnections.Apply(b, x, attentionPre, hc, hidden, $"{p}mlp_hc.input"), b.Weight(p + "post_attention_norm.weight"), _hp.Eps,
            $"{p}post_attn_norm");
        var ffn = FeedForward(b, h, p);
        return StreamHyperConnections.Merge(b, x, ResidualBranch(b, ffn), post, mixing, hc, hidden, $"{p}l_out");
    }

    // The layer with block attention residuals: x is the running sum of the current block; each sub-layer reads the mix of the banked
    // checkpoints and that sum, and a layer that starts a block banks its input and restarts the sum from its mixer's output.
    private Value ResidualAttentionLayer(GraphBuilder b, Value x, string p, int il, Value headIds, Func<Value, Value> rotate, Value? temperature,
        AttentionResiduals residuals)
    {
        var input = residuals.Mix(b, x, p + "attention_res.score.weight", _hp.HiddenSize, $"{p}attn_res");
        bool restart = residuals.StartsBlock(il);
        if (restart)
        {
            residuals.Push(b, x, _hp.HiddenSize);
        }

        var h = b.RmsNorm(input, b.Weight(p + "input_norm.weight"), _hp.Eps, $"{p}attn_norm");
        var attention = ResidualBranch(b, Mixer(b, h, p, il, headIds, rotate, temperature));
        x = restart ? attention : b.Add(attention, x, $"{p}attn_residual");

        input = residuals.Mix(b, x, p + "mlp_res.score.weight", _hp.HiddenSize, $"{p}mlp_res");
        h = b.RmsNorm(input, b.Weight(p + "post_attention_norm.weight"), _hp.Eps, $"{p}post_attn_norm");
        var ffn = FeedForward(b, h, p);
        return b.Add(ResidualBranch(b, ffn), x, $"{p}l_out");
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
        => _indexer!.Options.RopeStyle == "none" ? t => t : RopeOf(b, positions, _indexer.Options.RopeStyle, _hp.Attention.RopeMagnitude);

    /// <summary>The model's rotary embedding of the rope parts at given positions, with a base and optional frequency factors.</summary>
    private Func<Value, Value, Value> RopeAt(GraphBuilder b, float ropeBase, string factorsName)
    {
        var factors = b.TryWeight(factorsName);
        var attributes = Attributes.From(("dims", _hp.Attention.RopeDims), ("base", ropeBase), ("style", _hp.RopeStyle));
        return (t, positions) => b.Emit1("Rope", factors is null ? [t, positions] : [t, positions, factors], attributes);
    }

    private Func<Value, Value> RopeOf(GraphBuilder b, Value positions, string style, float magnitude, float? ropeBase = null,
        string factorsName = "rope.freq_factors")
    {
        var factors = b.TryWeight(factorsName);
        if (_longRope && b.TryWeight("rope.freq_factors_long") is { } longFactors)
        {
            factors = longFactors;
        }

        var attributes = Attributes.From(("dims", _hp.Attention.RopeDims), ("base", ropeBase ?? _hp.RopeBase), ("style", style));
        if (magnitude != 1f)
        {
            attributes = attributes.With("attn_factor", magnitude);
        }

        return t => b.Emit1("Rope", factors is null ? [t, positions] : [t, positions, factors], attributes);
    }
}
