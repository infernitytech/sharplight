using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Layer-pattern hybrid decoder (canonical arch <c>hybrid</c>): a stack of residual layers whose token mixer is a Mamba-2 state-space layer, grouped-query
/// attention, or both in parallel (their outputs are summed), optionally followed by a gated feed-forward block (dense and/or routed experts).
/// Covers MiniMax-01 (lightning attention interleaved with attention), Jamba (Mamba-1 / attention / MLP), Granite-4.0-H (interleaved Mamba-2 / attention, dense or shared-expert MoE feed-forward), Falcon-H1 (attention and Mamba-2 in every layer), LFM2 (gated short convolutions interleaved with attention),
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
/// <c>logit_scale</c> (multiplier of the logits, default 1), <c>residual_scale</c> (default 1), <c>mlp.activation</c>, <c>attention.*</c>/<c>rope.*</c>, <c>ssm.*</c>, <c>moe.*</c>.
/// Weights: <c>token_embedding.weight</c>, <c>output_norm.weight</c>, <c>lm_head.weight</c> (optional, tied otherwise); per layer <c>input_norm.weight</c> and,
/// with a feed-forward, <c>post_attention_norm.weight</c>.
/// </remarks>
public sealed class HybridAdapter : IModelAdapter
{
    /// <inheritdoc/>
    public string Name => "Hybrid (Mamba-2 / attention / gated feed-forward layer stack)";
    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => ["hybrid"];

    private sealed record LayerStates(StateDefinition? Conv, StateDefinition? Scan, StateDefinition? Keys, StateDefinition? Values, StateDefinition? ShortConv,
        StateDefinition? Lightning);

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        int layerCount = src.Arch<int>("layer_count");
        int hidden = src.Arch<int>("hidden_size");
        float eps = src.Arch("norm_eps", 1e-5f);
        float embeddingScale = src.Arch("embedding_scale", 1f);
        float logitScale = src.Arch("logit_scale", 1f);
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
            var kv = src.Weight(p + "attention.q.weight") is not null ? attention.DefineStates($"layers.{il}", options.KvCacheType) : default;
            var conv = src.Weight(p + "shortconv.in.weight") is { } convIn ? shortConv.DefineState($"layers.{il}", (int)(convIn.Shape[0].Fixed / 3)) : null;
            var linear = LightningAttention.IsPresent(src.Weight, p) ? lightning.DefineState($"layers.{il}") : null;
            layers[il] = new LayerStates(ssm.Conv, ssm.Scan, kv.Keys, kv.Values, conv, linear);
            states.AddRange(new[] { ssm.Conv, ssm.Scan, kv.Keys, kv.Values, conv, linear }.OfType<StateDefinition>());
        }

        var graphs = new Dictionary<string, ComputeGraph> { [WellKnown.DecodeGraph] = BuildTrunk(WellKnown.DecodeGraph, false) };
        if (options.IncludeAuxiliaryGraphs)
        {
            graphs["decode_embeddings"] = BuildTrunk("decode_embeddings", true);
        }

        return RecurrentModelDefinition.Create(src, layerCount, hidden, graphs, states);

        ComputeGraph BuildTrunk(string name, bool embeddingsInput)
        {
            var b = new GraphBuilder(name, src.Weight);
            var T = Dim.Sym(WellKnown.TokenSymbol);
            Value x;
            using (b.Group("input"))
            {
                if (embeddingsInput)
                {
                    x = b.Input("embeddings", InputRole.Embeddings, DataType.F32, new Shape(T, hidden));
                }
                else
                {
                    x = b.Gather(b.Weight("token_embedding.weight"), b.Input("tokens", InputRole.TokenIds, DataType.I32, new Shape(T)), "inp_embd");
                    if (embeddingScale != 1f)
                    {
                        x = b.Scale(x, embeddingScale);
                    }
                }
            }

            var positions = b.Input("positions", InputRole.Positions, DataType.I32, new Shape(T, 1));
            var outRows = b.Input("output_rows", InputRole.OutputRows, DataType.I32, new Shape(Dim.Sym("R")));
            for (int il = 0; il < layerCount; il++)
            {
                using var _ = b.Group($"layers.{il}");
                x = Layer(b, x, positions, il);
            }

            using (b.Group("output"))
            {
                var hiddenStates = b.RmsNorm(x, b.Weight("output_norm.weight"), eps, "result_norm");
                var rows = b.Gather(hiddenStates, outRows, "selected_rows");
                var head = b.TryWeight("lm_head.weight") ?? b.Weight("token_embedding.weight");
                var logits = b.MatMul(rows, head, "logits");
                b.Output("logits", logitScale != 1f ? b.Scale(logits, logitScale) : logits);
                b.Output("hidden", hiddenStates);
            }

            return b.Build();
        }

        Value Layer(GraphBuilder b, Value x, Value positions, int il)
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
                var attended = attention.Build(b, h, positions, il, p, layer.Keys, layer.Values!);
                mixed = mixed is null ? attended : b.Add(mixed, attended);
            }

            var mixer = mixed ?? throw new InvalidDataException($"hybrid: layer {il} has neither a state-space nor an attention mixer.");
            var mixerOut = Scaled(b, OutputNorm(b, mixer, p + "post_mixer_norm.weight"), PerLayer(mixerScale, il, residualScale));
            x = b.Add(mixerOut, Scaled(b, residualFromNorm ? h : x, PerLayer(mixerInputScale, il, 1f)), $"{p}mix_out");
            if (!HybridFeedForward.Exists(src, p))
            {
                return x;
            }

            var normed = b.RmsNorm(x, b.Weight(p + "post_attention_norm.weight"), eps, $"{p}post_attn_norm");
            var ffnOut = Scaled(b, OutputNorm(b, feedForward.Build(b, normed, p), p + "mlp.out_norm.weight"), PerLayer(ffnScale, il, residualScale));
            return b.Add(ffnOut, Scaled(b, residualFromNorm ? normed : x, PerLayer(ffnInputScale, il, 1f)), $"{p}l_out");
        }

        static float PerLayer(float[] scales, int il, float fallback) => il < scales.Length ? scales[il] : fallback;

        Value OutputNorm(GraphBuilder b, Value v, string weightName) => b.TryWeight(weightName) is { } weight ? b.RmsNorm(v, weight, eps) : v;

        static Value Scaled(GraphBuilder b, Value v, float scale) => scale == 1f ? v : b.Scale(v, scale);
    }
}
