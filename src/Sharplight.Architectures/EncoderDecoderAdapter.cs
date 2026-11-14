using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Pre-norm transformer encoder-decoder (canonical arch <c>encoder_decoder</c>): T5, T5 v1.1 / Flan-T5, mT5 and encoder-only T5 checkpoints.
/// Builds <see cref="WellKnown.EncodeGraph"/> (the bidirectional encoder over the whole source sequence),
/// <see cref="WellKnown.DecodeGraph"/> (the causal decoder with a KV cache for its self-attention and cross-attention over the
/// <see cref="InputRole.EncoderOutput"/> input, whose keys and values are projected from the encoder output on every step) and the
/// packed-sequence <see cref="WellKnown.EmbedGraph"/> over the encoder (sentence embeddings, T5 text encoders). A model without decoder
/// layers only gets the embedding graph.
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>encoder_decoder.*</c>): <c>hidden_size</c>, <c>encoder.layer_count</c>, <c>decoder.layer_count</c> (0: encoder only),
/// <c>attention.head_count</c>, <c>attention.head_dim</c> (default hidden_size / heads; T5's d_kv), <c>attention.scale</c> (default 1, T5
/// folds the score scale into its query projection), <c>relative_attention.max_distance</c> (default 128), <c>norm</c> ("rms", T5's
/// default, or "layer"), <c>norm_eps</c> (default 1e-6), <c>mlp.activation</c> ("relu" (default), "gelu", "gelu_tanh", "silu"),
/// <c>output.scale</c> (factor of the final hidden state before the LM head; T5 with tied embeddings: hidden_size^-0.5),
/// <c>decoder_start_token</c> (default 0), <c>context_length</c> (default 512).
/// <para>Weights: <c>token_embedding.weight</c> [V, H] (shared by encoder and decoder), optional <c>output.weight</c> [V, H] (default: the
/// embedding); per encoder layer <c>encoder.layers.{i}.</c> and decoder layer <c>decoder.layers.{i}.</c>: <c>attention_norm.weight</c>,
/// <c>attention.{q,k,v,o}.weight</c>, <c>mlp_norm.weight</c>, <c>mlp.up.weight</c>, optional <c>mlp.gate.weight</c> (gated: act(gate) · up),
/// <c>mlp.down.weight</c>; per decoder layer also <c>cross_attention_norm.weight</c> and <c>cross_attention.{q,k,v,o}.weight</c>. Every
/// projection and norm may have a <c>.bias</c>. <c>encoder.attention.relative_bias.weight</c> / <c>decoder.attention.relative_bias.weight</c>
/// [buckets, heads] (f32, f16 or bf16): T5's bucketed relative position bias, shared by all layers of the stack (a layer's own
/// <c>{stack}.layers.{i}.attention.relative_bias.weight</c> takes precedence; absent: no position information);
/// <c>encoder.output_norm.weight</c>, <c>decoder.output_norm.weight</c>.</para>
/// </remarks>
public sealed class EncoderDecoderAdapter : IModelAdapter
{
    /// <inheritdoc/>
    public string Name => "Transformer encoder-decoder";
    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => ["encoder_decoder"];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        int nEmbd = src.Arch<int>("hidden_size");
        int encoderLayers = src.Arch<int>("encoder.layer_count");
        int decoderLayers = src.Arch("decoder.layer_count", 0);
        int nHead = src.Arch<int>("attention.head_count");
        int headDim = src.Arch("attention.head_dim", nEmbd / nHead);
        int inner = nHead * headDim;
        float scale = src.Arch("attention.scale", 1f);
        int maxDistance = src.Arch("relative_attention.max_distance", 128);
        string norm = src.Arch("norm", "rms");
        float eps = src.Arch("norm_eps", 1e-6f);
        string act = src.Arch("mlp.activation", "relu");
        float outputScale = src.Arch("output.scale", 1f);
        int ctx = src.Arch("context_length", 512);
        var encoderBias = StackBias("encoder", encoderLayers, bidirectional: true);
        var decoderBias = StackBias("decoder", decoderLayers, bidirectional: false);
        string pooling = PackedSequences.NormalizePooling(options.EmbeddingPooling) ?? PackedSequences.NormalizePooling(src.Get<string?>(CanonicalKeys.EmbeddingPooling, null)) ?? "mean";
        bool normalize = options.EmbeddingNormalize ?? src.Get(CanonicalKeys.EmbeddingNormalize, true);

        var states = new List<StateDefinition>();
        var kCache = new StateDefinition[decoderLayers];
        var vCache = new StateDefinition[decoderLayers];
        for (int il = 0; il < decoderLayers; il++)
        {
            string g = $"decoder.layers.{il}";
            var shape = new Shape(Dim.Sym(WellKnown.ContextSymbol), nHead, headDim);
            states.Add(kCache[il] = new StateDefinition { Name = $"{g}.k_cache", Type = options.KvCacheTypeFor(headDim), Shape = shape, Kind = StateKind.KeyCache, Group = g });
            states.Add(vCache[il] = new StateDefinition { Name = $"{g}.v_cache", Type = options.KvCacheTypeFor(headDim), Shape = shape, Kind = StateKind.ValueCache, Group = g });
        }

        var graphs = new Dictionary<string, ComputeGraph> { [WellKnown.EmbedGraph] = BuildEmbed() };
        if (decoderLayers > 0)
        {
            graphs[WellKnown.EncodeGraph] = BuildEncode();
            graphs[WellKnown.DecodeGraph] = BuildDecode();
        }

        var props = new Dictionary<string, object>
        {
            [WellKnown.VocabSize] = (int)src.Weight("token_embedding.weight")!.Shape[0].Fixed,
            [WellKnown.ContextLength] = ctx,
            [WellKnown.EmbeddingLength] = nEmbd,
            [WellKnown.PositionAxes] = 1,
            [WellKnown.EmbeddingPooling] = pooling,
            [WellKnown.EmbeddingNormalize] = normalize,
            [WellKnown.CausalAttention] = false,
            [WellKnown.DecoderStartToken] = src.Arch("decoder_start_token", 0),
            ["n_layer"] = encoderLayers + decoderLayers,
        };

        return new ModelDefinition
        {
            Architecture = src.Architecture,
            Name = src.Get(CanonicalKeys.Name, src.Architecture),
            Weights = src.Weights,
            Graphs = graphs,
            States = states,
            SourceMetadata = src.Metadata,
            OwnedResources = src.OwnedResources,
            Properties = props,
        };

        ComputeGraph BuildEncode()
        {
            var b = new GraphBuilder(WellKnown.EncodeGraph, src.Weight);
            Value x;
            using (b.Group("input"))
            {
                var tokens = b.Input("tokens", InputRole.TokenIds, DataType.I32, new Shape(Dim.Sym(WellKnown.TokenSymbol)));
                x = b.Gather(b.Weight("token_embedding.weight"), tokens, "inp_embd");
            }

            x = EncoderLayers(b, x, (q, k, v, il, name) => b.Emit1("Attention", [q, k, v], Attributes.From([("scale", scale), ("causal", false), .. BiasAttributes(encoderBias[il])]), name));
            using (b.Group("output"))
            {
                b.Output(WellKnown.EncoderOutput, Norm(b, x, "encoder.output_norm", "encoder_output"));
            }

            return b.Build();
        }

        ComputeGraph BuildEmbed()
        {
            var b = new GraphBuilder(WellKnown.EmbedGraph, src.Weight);
            var inp = PackedSequences.Declare(b, tokenTypes: false);
            Value x;
            using (b.Group("input"))
            {
                x = b.Gather(b.Weight("token_embedding.weight"), inp.Tokens, "inp_embd");
            }

            x = EncoderLayers(b, x, (q, k, v, il, name) => PackedSequences.Attention(b, q, k, v, inp.RowSpans, scale, causal: false, name: name, positionBias: encoderBias[il]));
            using (b.Group("output"))
            {
                PackedSequences.EmbeddingsOutput(b, Norm(b, x, "encoder.output_norm", "encoder_output"), inp.SequenceSpans, pooling, normalize);
            }

            return b.Build();
        }

        Value EncoderLayers(GraphBuilder b, Value x, Func<Value, Value, Value, int, string, Value> attention)
        {
            for (int il = 0; il < encoderLayers; il++)
            {
                using var _ = b.Group($"encoder.layers.{il}");
                string p = $"encoder.layers.{il}.";
                var h = Norm(b, x, p + "attention_norm");
                var attn = attention(Heads(b, Linear(b, h, p + "attention.q")), Heads(b, Linear(b, h, p + "attention.k")), Heads(b, Linear(b, h, p + "attention.v")), il, $"{p}attn");
                x = b.Add(x, Linear(b, b.Reshape(attn, 0, inner), p + "attention.o", $"{p}attn_out"));
                x = b.Add(x, FeedForward(b, Norm(b, x, p + "mlp_norm"), p));
            }

            return x;
        }

        ComputeGraph BuildDecode()
        {
            var b = new GraphBuilder(WellKnown.DecodeGraph, src.Weight);
            var T = Dim.Sym(WellKnown.TokenSymbol);
            Value x;
            Value encoded;
            using (b.Group("input"))
            {
                var tokens = b.Input("tokens", InputRole.TokenIds, DataType.I32, new Shape(T));
                encoded = b.Input(WellKnown.EncoderOutput, InputRole.EncoderOutput, DataType.F32, new Shape(Dim.Sym(WellKnown.EncoderSymbol), nEmbd));
                x = b.Gather(b.Weight("token_embedding.weight"), tokens, "inp_embd");
            }

            var outRows = b.Input("output_rows", InputRole.OutputRows, DataType.I32, new Shape(Dim.Sym("R")));
            for (int il = 0; il < decoderLayers; il++)
            {
                using var _ = b.Group($"decoder.layers.{il}");
                string p = $"decoder.layers.{il}.";
                var h = Norm(b, x, p + "attention_norm");
                var kc = b.State(kCache[il]);
                var vc = b.State(vCache[il]);
                b.CacheWrite(kc, Heads(b, Linear(b, h, p + "attention.k")));
                b.CacheWrite(vc, Heads(b, Linear(b, h, p + "attention.v")));
                var attn = b.Emit1("CachedAttention", [Heads(b, Linear(b, h, p + "attention.q")), kc, vc],
                    Attributes.From([("scale", scale), ("offset", WellKnown.PastScalar), .. BiasAttributes(decoderBias[il])]), $"{p}attn");
                x = b.Add(x, Linear(b, b.Reshape(attn, 0, inner), p + "attention.o", $"{p}attn_out"));

                // Cross-attention: keys and values projected from the encoder output (recomputed every step; the source is short).
                h = Norm(b, x, p + "cross_attention_norm");
                var cross = b.Emit1("Attention",
                    [Heads(b, Linear(b, h, p + "cross_attention.q")), Heads(b, Linear(b, encoded, p + "cross_attention.k")), Heads(b, Linear(b, encoded, p + "cross_attention.v"))],
                    Attributes.From(("scale", scale), ("causal", false)), $"{p}cross_attn");
                x = b.Add(x, Linear(b, b.Reshape(cross, 0, inner), p + "cross_attention.o", $"{p}cross_attn_out"));
                x = b.Add(x, FeedForward(b, Norm(b, x, p + "mlp_norm"), p));
            }

            using (b.Group("output"))
            {
                var hidden = Norm(b, x, "decoder.output_norm", "result_norm");
                var rows = b.Gather(hidden, outRows, "selected_rows");
                if (outputScale != 1f)
                {
                    rows = b.Scale(rows, outputScale);
                }

                string head = b.HasWeight("output.weight") ? "output.weight" : "token_embedding.weight";
                b.Output(WellKnown.LogitsOutput, b.Linear(rows, head, name: "logits"));
                b.Output(WellKnown.HiddenOutput, hidden);
            }

            return b.Build();
        }

        static (string, object)[] BiasAttributes(RelativePositionBias? bias) => bias?.Attributes ?? [];

        // Per layer: the layer's own bucket table, else the stack's shared one (T5: only the first layer has one, used by all).
        RelativePositionBias?[] StackBias(string stack, int layers, bool bidirectional)
        {
            var shared = src.Weight($"{stack}.attention.relative_bias.weight") is { } w ? RelativePositionBias.FromBuckets(w, maxDistance, bidirectional) : null;
            var result = new RelativePositionBias?[layers];
            for (int il = 0; il < layers; il++)
            {
                result[il] = src.Weight($"{stack}.layers.{il}.attention.relative_bias.weight") is { } own
                    ? RelativePositionBias.FromBuckets(own, maxDistance, bidirectional)
                    : shared;
            }

            return result;
        }

        Value Heads(GraphBuilder b, Value v) => b.Reshape(v, 0, nHead, headDim);

        Value Linear(GraphBuilder b, Value v, string module, string? name = null)
            => b.Linear(v, module + ".weight", b.HasWeight(module + ".bias") ? module + ".bias" : null, name);

        Value Norm(GraphBuilder b, Value v, string module, string? name = null) => norm switch
        {
            "rms" => b.RmsNorm(v, b.Weight(module + ".weight"), eps, name),
            "layer" => b.Emit1("LayerNorm", [v, b.Weight(module + ".weight"), b.Weight(module + ".bias")], Attributes.From(("eps", eps)), name),
            _ => throw new NotSupportedException($"encoder_decoder: unknown norm '{norm}'."),
        };

        Value FeedForward(GraphBuilder b, Value h, string p)
        {
            var up = Linear(b, h, p + "mlp.up");
            var y = b.HasWeight(p + "mlp.gate.weight") ? b.Mul(Act(b, Linear(b, h, p + "mlp.gate")), up) : Act(b, up);
            return Linear(b, y, p + "mlp.down", $"{p}ffn_out");
        }

        Value Act(GraphBuilder b, Value t) => act switch
        {
            "relu" => b.Unary("Relu", t),
            "gelu" => b.Unary("Gelu", t),
            "gelu_tanh" => b.Unary("GeluTanh", t),
            "silu" => b.Silu(t),
            _ => throw new NotSupportedException($"encoder_decoder: unknown mlp.activation '{act}'."),
        };
    }
}
