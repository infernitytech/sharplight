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
    private readonly IMlpActivation _activation;
    private readonly StateDefinition[] _caches;
    private readonly bool _longRope;
    private readonly Func<string, WeightInfo?>? _draft;
    private readonly StateDefinition? _draftCache;

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
        _caches = [.. Enumerable.Range(0, _hp.LayerCount).Select(il => _attention.CreateCache($"layers.{il}", options.KvCacheType))];
        // The NextN draft head (DeepSeek-V3: one MTP layer after the trunk's, see MtpDraftHead.FirstNextnModule) with its own cache.
        _draft = MtpDraftHead.FirstNextnModule(_src.Weight);
        if (_draft is not null)
        {
            _draftCache = _attention.CreateCache("mtp.layers.0", options.KvCacheType);
        }

        IncludeEmbeddingsGraph = options.IncludeAuxiliaryGraphs;
    }

    /// <summary>The persistent states (one latent cache per layer, and the draft layer's).</summary>
    public List<StateDefinition> States => _draftCache is null ? [.. _caches] : [.. _caches, _draftCache];

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
                (b, x, j, positions) => Layer(b, x, $"mtp.layers.{j}.", _draftCache!, LatentAttention.HeadIndices(b, positions, _hp.Attention.Heads), Rotation(b, positions),
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
        for (int il = 0; il < _hp.LayerCount; il++)
        {
            using var _ = b.Group($"layers.{il}");
            x = Layer(b, x, $"layers.{il}.", _caches[il], headIds, rotate, temperature);
        }

        using (b.Group("output"))
        {
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

    // One decoder layer: weights below p, latent cache `cache`.
    private Value Layer(GraphBuilder b, Value x, string p, StateDefinition cache, Value headIds, Func<Value, Value> rotate, Value? temperature)
    {
        var h = b.RmsNorm(x, b.Weight(p + "input_norm.weight"), _hp.Eps, $"{p}attn_norm");
        var attention = _attention.Build(b, h, p, cache, headIds, rotate, temperature);
        x = b.Add(ResidualBranch(b, attention), x, $"{p}attn_residual");

        h = b.RmsNorm(x, b.Weight(p + "post_attention_norm.weight"), _hp.Eps, $"{p}post_attn_norm");
        var ffn = MoeBlock.IsPresent(b, p + "mlp.") ? Experts(b, h, p) : Mlp(b, h, p, "mlp.", $"{p}ffn_out");
        return b.Add(ResidualBranch(b, ffn), x, $"{p}l_out");
    }

    /// <summary>The query temperature of every row (Mistral 4 / Mistral Large 3), or null when the model has none.</summary>
    private Value? Temperature(GraphBuilder b, Value positions)
        => _hp.TemperatureScale == 0f ? null : AttentionTemperature.Build(b, positions, _hp.TemperatureScale, _hp.TemperatureLength);

    private Value ResidualBranch(GraphBuilder b, Value branch) => _hp.ResidualScale == 1f ? branch : b.Scale(branch, _hp.ResidualScale);

    /// <summary>The rotary embedding of the rope parts ([T, heads, rope_dim]), with the frequency factors the context selects.</summary>
    private Func<Value, Value> Rotation(GraphBuilder b, Value positions)
    {
        var factors = b.TryWeight("rope.freq_factors");
        if (_longRope && b.TryWeight("rope.freq_factors_long") is { } longFactors)
        {
            factors = longFactors;
        }

        var attributes = Attributes.From(("dims", _hp.Attention.RopeDims), ("base", _hp.RopeBase), ("style", _hp.RopeStyle));
        return t => b.Emit1("Rope", factors is null ? [t, positions] : [t, positions, factors], attributes);
    }
}
