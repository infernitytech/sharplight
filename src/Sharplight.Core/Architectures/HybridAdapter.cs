using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;
using Sharplight.Vision;

namespace Sharplight.Architectures;

/// <summary>
/// Layer-pattern hybrid decoder (canonical arch <c>hybrid</c>): a stack of residual layers whose token mixer is a Mamba-2 state-space layer, grouped-query
/// attention, or both in parallel (their outputs are summed), optionally followed by a gated feed-forward block (dense and/or routed experts).
/// Covers ZAYA1 (CCA attention and a skip-expert MoE in every layer), MiniMax-01 (lightning attention interleaved with attention), Jamba (Mamba-1 / attention / MLP), Granite-4.0-H (interleaved Mamba-2 / attention, dense or shared-expert MoE feed-forward), Falcon-H1 (attention and Mamba-2 in every layer), LFM2 (gated short convolutions interleaved with attention),
/// PLaMo-2 (Mamba-2 style layers with their own projection layout, see <see cref="Plamo2Layer"/>, interleaved with attention, sandwich norms around both blocks),
/// and pure Mamba-2 stacks (no feed-forward).
/// </summary>
/// <remarks>
/// The layer pattern is read from the weights: a layer with <c>ssm.in.weight</c> has a state-space mixer (Mamba-1, <see cref="Mamba1Layer"/>, when it also has <c>ssm.x.weight</c>, else Mamba-2), one with <c>shortconv.in.weight</c> the short-convolution mixer (<see cref="ShortConvLayer"/>), one with <c>attention.q.weight</c> has attention;
/// a layer with <c>mlp.*</c> weights has the feed-forward block (see <see cref="Mamba2Layer"/>, <see cref="HybridAttention"/>,
/// <see cref="HybridFeedForward"/> for their weight vocabulary).
/// A layer with <c>lightning.qkv.weight</c> has lightning attention (<see cref="LightningAttention"/>, MiniMax-01).
/// Layer: <c>x = α · r + β · mixers(norm(x))</c>, then (with a feed-forward) <c>x = α′ · r′ + β′ · ffn(post_attention_norm(x))</c>, where the residual
/// r is the layer input x, or the normalized input when <c>residual.from_norm</c> is set (MiniMax-01; likewise r′); α, β, α′, β′ come per layer from
/// <c>residual.mixer_input_scale</c>, <c>residual.mixer_scale</c>, <c>residual.ffn_input_scale</c>, <c>residual.ffn_scale</c> (α defaults to 1,
/// β to <c>residual_scale</c>); an optional
/// <c>post_mixer_norm.weight</c> / <c>mlp.out_norm.weight</c> normalizes the mixer / feed-forward output before it is added (PLaMo-2).
/// Hyper-parameters (<c>hybrid.*</c>): <c>layer_count</c>, <c>hidden_size</c>, <c>norm_eps</c>, <c>context_length</c>, optional <c>embedding_scale</c> (default 1),
/// <c>logit_scale</c> (multiplier of the logits, default 1), <c>logits.suppressed_ranges</c> (int[] of [start, end) pairs of token ids never emitted),
/// <c>residual_scale</c> (default 1), <c>mlp.activation</c>, <c>attention.*</c>/<c>rope.*</c>, <c>ssm.*</c>, <c>moe.*</c>.
/// Weights: <c>token_embedding.weight</c>, <c>output_norm.weight</c>, <c>lm_head.weight</c> (optional, tied otherwise); per layer <c>input_norm.weight</c> and,
/// with a feed-forward, <c>post_attention_norm.weight</c>.
/// Optional (Inkling): <c>embedding_norm.weight</c> RMS-normalizes the token embeddings; <c>post_mixer_conv.weight</c> / <c>mlp.out_conv.weight</c>
/// [hidden, K] f32 add a causal depthwise convolution of the mixer / feed-forward output to itself before it joins the residual
/// (<c>y + conv(y)</c>, states <c>{group}.mixer_conv_state</c> / <c>ffn_conv_state</c>); <c>mlp.output_scale</c> [1] multiplies the
/// feed-forward output.
/// Optional (ZAYA1): a layer with <c>attention.cca.conv.weight</c> mixes with compressed convolutional attention (<see cref="CcaAttention"/>);
/// <c>mlp.router.out.weight</c> selects the MLP router with a skip expert (<see cref="SkipExpertRouter"/>, its router rows carried from layer to
/// layer); per-channel residual scales <c>residual.{mixer,ffn}_{out_scale,in_scale,bias}</c> [hidden] replace the scalar ones
/// (<c>x = out · out_scale + r · in_scale + bias</c>); <c>input_scale.weight</c> / <c>input_scale.bias</c> [hidden] turn every input row x into
/// <c>(x + bias) · scale</c>.
/// Optional vision tower (image input, LFM2-VL): hyper-parameters <c>hybrid.vision.*</c> and weights <c>vision.*</c> as documented on
/// <see cref="VisionParameters"/>, in the primary source or an auxiliary <see cref="AuxiliaryRoles.Mmproj"/> source; the definition then has the
/// <see cref="WellKnown.VisionEncodeGraph"/> and <see cref="WellKnown.DecodeMultimodalGraph"/> graphs (token or image rows per prompt row) and prompts
/// are assembled by <see cref="Qwen35Multimodal"/> on one position axis.
/// </remarks>
public sealed class HybridAdapter : IModelAdapter
{
    /// <inheritdoc/>
    public string Name => "Hybrid (Mamba-2 / attention / gated feed-forward layer stack)";
    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => ["hybrid"];

    private sealed record LayerStates(StateDefinition? Conv, StateDefinition? Scan, StateDefinition? Keys, StateDefinition? Values, StateDefinition? ShortConv,
        StateDefinition? Lightning, StateDefinition? KeyConv = null, StateDefinition? ValueConv = null, StateDefinition? MixerConv = null,
        StateDefinition? FeedForwardConv = null, (StateDefinition QueryKey, StateDefinition Depthwise, StateDefinition Value)? Cca = null);

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        int layerCount = src.Arch<int>("layer_count");
        int hidden = src.Arch<int>("hidden_size");
        float eps = src.Arch("norm_eps", 1e-5f);
        float embeddingScale = src.Arch("embedding_scale", 1f);
        float logitScale = src.Arch("logit_scale", 1f);
        int[] suppressedRanges = src.Arch("logits.suppressed_ranges", Array.Empty<int>());
        float residualScale = src.Arch("residual_scale", 1f);
        bool residualFromNorm = src.Arch("residual.from_norm", false);
        float[] mixerInputScale = src.Arch("residual.mixer_input_scale", Array.Empty<float>());
        float[] mixerScale = src.Arch("residual.mixer_scale", Array.Empty<float>());
        float[] ffnInputScale = src.Arch("residual.ffn_input_scale", Array.Empty<float>());
        float[] ffnScale = src.Arch("residual.ffn_scale", Array.Empty<float>());
        var mamba = new Mamba2Layer(src, eps);
        var mamba1 = new Mamba1Layer(src, eps);
        var plamo = new Plamo2Layer(src, eps);
        var attention = new HybridAttention(src);
        var cca = new CcaAttention(attention);
        var shortConv = new ShortConvLayer(src);
        var lightning = new LightningAttention(src);
        var feedForward = new HybridFeedForward(src, hidden);

        var states = new List<StateDefinition>();
        var layers = new LayerStates[layerCount];
        for (int il = 0; il < layerCount; il++)
        {
            string p = $"layers.{il}.";
            var ssm = src.Weight(p + "ssm.in.weight") is null ? default
                : Plamo2Layer.Is(src, p) ? plamo.DefineStates(src, $"layers.{il}", p)
                : Mamba1Layer.Is(src, p) ? mamba1.DefineStates(src, $"layers.{il}", p) : mamba.DefineStates($"layers.{il}");
            var kv = src.Weight(p + "attention.q.weight") is not null ? attention.DefineStates($"layers.{il}", options.KvCacheType, il) : default;
            var kvConv = HybridAttention.DefineConvStates(src, $"layers.{il}", p);
            var mixerConv = OutputConvState(src, $"layers.{il}", "mixer_conv_state", p + "post_mixer_conv.weight");
            var ffnConv = OutputConvState(src, $"layers.{il}", "ffn_conv_state", p + "mlp.out_conv.weight");
            var conv = src.Weight(p + "shortconv.in.weight") is { } convIn ? shortConv.DefineState($"layers.{il}", (int)(convIn.Shape[0].Fixed / 3)) : null;
            var linear = LightningAttention.IsPresent(src.Weight, p) ? lightning.DefineState($"layers.{il}") : null;
            var ccaStates = CcaAttention.Is(src, p) ? cca.DefineStates($"layers.{il}", il) : default((StateDefinition QueryKey, StateDefinition Depthwise, StateDefinition Value)?);
            layers[il] = new LayerStates(ssm.Conv, ssm.Scan, kv.Keys, kv.Values, conv, linear, kvConv.Keys, kvConv.Values, mixerConv, ffnConv, ccaStates);
            states.AddRange(new[] { ssm.Conv, ssm.Scan, kv.Keys, kv.Values, conv, linear, kvConv.Keys, kvConv.Values, mixerConv, ffnConv }.OfType<StateDefinition>());
            if (ccaStates is { } windows)
            {
                states.AddRange([windows.QueryKey, windows.Depthwise, windows.Value]);
            }
        }

        // Vision tower: in the primary source or an auxiliary "mmproj" source, same canonical names either way.
        var visionSrc = sources.Auxiliary.GetValueOrDefault(AuxiliaryRoles.Mmproj) ?? src;
        var vision = VisionParameters.FromMetadata(visionSrc.Metadata, visionSrc.Architecture + ".vision.");
        if (vision is not null && !string.Equals(visionSrc.Architecture, src.Architecture, StringComparison.OrdinalIgnoreCase))
        {
            throw new NotSupportedException($"The vision tower source '{visionSrc.Path}' belongs to architecture '{visionSrc.Architecture}', not '{src.Architecture}'.");
        }

        if (vision is not null && (vision.DeepstackLayers.Length > 0 || vision.OutHiddenSize != hidden || vision.Sam is not null))
        {
            throw new NotSupportedException($"hybrid: the vision tower must produce {hidden}-wide rows without DeepStack blocks or SAM stages.");
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
        var graphs = new Dictionary<string, ComputeGraph> { [WellKnown.DecodeGraph] = BuildTrunk(WellKnown.DecodeGraph, false) };
        if (options.IncludeAuxiliaryGraphs)
        {
            graphs["decode_embeddings"] = BuildTrunk("decode_embeddings", true);
        }

        if (vision is not null)
        {
            graphs[WellKnown.DecodeMultimodalGraph] = BuildTrunk(WellKnown.DecodeMultimodalGraph, true, mixed: true);
            graphs[WellKnown.VisionEncodeGraph] = VisionTowerGraph.Build(vision, Lookup);
        }

        return RecurrentModelDefinition.Create(src, layerCount, hidden, graphs, states, allWeights, vision, visionSrc);

        ComputeGraph BuildTrunk(string name, bool embeddingsInput, bool mixed = false)
        {
            var b = new GraphBuilder(name, src.Weight);
            var T = Dim.Sym(WellKnown.TokenSymbol);
            Value x;
            using (b.Group("input"))
            {
                if (mixed)
                {
                    // Rows are token embeddings or media embeddings: x = tok·(1-mask) + emb·mask (the embedding scale and norm apply to token rows only).
                    var tokens = b.Input("tokens", InputRole.TokenIds, DataType.I32, new Shape(T));
                    var emb = b.Input("embeddings", InputRole.Embeddings, DataType.F32, new Shape(T, hidden));
                    var mask = b.Input("embedding_mask", InputRole.EmbeddingMask, DataType.F32, new Shape(T, 1));
                    var tok = TokenEmbeddings(b, tokens, "tok_embd");
                    x = b.Add(b.Mul(tok, b.Scale(mask, -1f, 1f)), b.Mul(emb, mask), "inp_embd");
                }
                else if (embeddingsInput)
                {
                    x = b.Input("embeddings", InputRole.Embeddings, DataType.F32, new Shape(T, hidden));
                }
                else
                {
                    x = TokenEmbeddings(b, b.Input("tokens", InputRole.TokenIds, DataType.I32, new Shape(T)), "inp_embd");
                }
            }

            if (b.TryWeight("input_scale.weight") is { } inputScale)
            {
                // ZAYA1: x = (x + bias) · scale on every input row.
                using var _ = b.Group("input");
                x = b.Mul(b.TryWeight("input_scale.bias") is { } inputBias ? b.Add(x, inputBias) : x, inputScale, "inp_scaled");
            }

            var positions = b.Input("positions", InputRole.Positions, DataType.I32, new Shape(T, 1));
            var outRows = b.Input("output_rows", InputRole.OutputRows, DataType.I32, new Shape(Dim.Sym("R")));
            Value? routerState = null;
            for (int il = 0; il < layerCount; il++)
            {
                using var _ = b.Group($"layers.{il}");
                x = Layer(b, x, positions, il, ref routerState);
            }

            using (b.Group("output"))
            {
                var hiddenStates = b.RmsNorm(x, b.Weight("output_norm.weight"), eps, "result_norm");
                var rows = b.Gather(hiddenStates, outRows, "selected_rows");
                var head = b.TryWeight("lm_head.weight") ?? b.Weight("token_embedding.weight");
                var logits = b.MatMul(rows, head, "logits");
                logits = logitScale != 1f ? b.Scale(logits, logitScale) : logits;
                b.Output("logits", SuppressedLogits.Apply(b, logits, checked((int)head.Shape[0].Fixed), suppressedRanges));
                b.Output("hidden", hiddenStates);
            }

            return b.Build();
        }

        Value Layer(GraphBuilder b, Value x, Value positions, int il, ref Value? routerState)
        {
            string p = $"layers.{il}.";
            var layer = layers[il];
            var h = b.RmsNorm(x, b.Weight(p + "input_norm.weight"), eps, $"{p}norm");
            Value? mixed = null;
            if (layer.Scan is not null)
            {
                mixed = Plamo2Layer.Is(src, p) ? plamo.Build(b, h, p, layer.Conv!, layer.Scan)
                    : Mamba1Layer.Is(src, p) ? mamba1.Build(b, h, p, layer.Conv!, layer.Scan) : mamba.Build(b, h, p, layer.Conv!, layer.Scan);
            }

            if (layer.ShortConv is not null)
            {
                mixed = shortConv.Build(b, h, p, layer.ShortConv);
            }

            if (layer.Lightning is not null)
            {
                mixed = lightning.Build(b, h, p, layer.Lightning);
            }

            if (layer.Keys is not null)
            {
                var attended = layer.Cca is { } windows
                    ? cca.Build(b, h, positions, il, p, layer.Keys, layer.Values!, windows)
                    : attention.Build(b, h, positions, il, p, layer.Keys, layer.Values!, layer.KeyConv, layer.ValueConv);
                mixed = mixed is null ? attended : b.Add(mixed, attended);
            }

            var mixer = HybridAttention.ResidualConv(b, mixed ?? throw new InvalidDataException($"hybrid: layer {il} has neither a state-space nor an attention mixer."),
                p + "post_mixer_conv.weight", layer.MixerConv);
            x = b.TryWeight(p + "residual.mixer_out_scale") is not null
                ? VectorResidual(b, mixer, x, p + "residual.mixer_", $"{p}mix_out")
                : b.Add(Scaled(b, OutputNorm(b, mixer, p + "post_mixer_norm.weight"), PerLayer(mixerScale, il, residualScale)),
                    Scaled(b, residualFromNorm ? h : x, PerLayer(mixerInputScale, il, 1f)), $"{p}mix_out");
            if (!HybridFeedForward.Exists(src, p))
            {
                return x;
            }

            var normed = b.RmsNorm(x, b.Weight(p + "post_attention_norm.weight"), eps, $"{p}post_attn_norm");
            var ffn = feedForward.Build(b, normed, p, ref routerState);
            if (b.TryWeight(p + "mlp.output_scale") is { } outputScale)
            {
                ffn = b.Mul(ffn, outputScale);
            }

            ffn = HybridAttention.ResidualConv(b, ffn, p + "mlp.out_conv.weight", layer.FeedForwardConv);
            if (b.TryWeight(p + "residual.ffn_out_scale") is not null)
            {
                return VectorResidual(b, ffn, x, p + "residual.ffn_", $"{p}l_out");
            }

            var ffnOut = Scaled(b, OutputNorm(b, ffn, p + "mlp.out_norm.weight"), PerLayer(ffnScale, il, residualScale));
            return b.Add(ffnOut, Scaled(b, residualFromNorm ? normed : x, PerLayer(ffnInputScale, il, 1f)), $"{p}l_out");
        }

        Value TokenEmbeddings(GraphBuilder b, Value tokens, string name)
        {
            var x = b.Gather(b.Weight("token_embedding.weight"), tokens, name);
            if (embeddingScale != 1f)
            {
                x = b.Scale(x, embeddingScale);
            }

            if (b.TryWeight("embedding_norm.weight") is { } embeddingNorm)
            {
                x = b.RmsNorm(x, embeddingNorm, eps, "inp_embd_norm");
            }

            return x;
        }

        // out · {prefix}out_scale + (residual · {prefix}in_scale + {prefix}bias), all [hidden] (ZAYA1).
        static Value VectorResidual(GraphBuilder b, Value output, Value residual, string prefix, string name)
        {
            var scaledResidual = b.Mul(residual, b.Weight(prefix + "in_scale"));
            if (b.TryWeight(prefix + "bias") is { } bias)
            {
                scaledResidual = b.Add(scaledResidual, bias);
            }

            return b.Add(b.Mul(output, b.Weight(prefix + "out_scale")), scaledResidual, name);
        }

        static float PerLayer(float[] scales, int il, float fallback) => il < scales.Length ? scales[il] : fallback;

        static StateDefinition? OutputConvState(ModelSource src, string group, string name, string weightName)
            => src.Weight(weightName) is { } w ? RecurrentStates.ConvWindow(group, name, (int)w.Shape[1].Fixed - 1, (int)w.Shape[0].Fixed) : null;

        Value OutputNorm(GraphBuilder b, Value v, string weightName) => b.TryWeight(weightName) is { } weight ? b.RmsNorm(v, weight, eps) : v;

        static Value Scaled(GraphBuilder b, Value v, float scale) => scale == 1f ? v : b.Scale(v, scale);
    }
}
