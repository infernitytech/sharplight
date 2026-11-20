using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Transformer encoder-decoder (canonical arch <c>encoder_decoder</c>): T5, T5 v1.1 / Flan-T5, mT5 / UMT5 and encoder-only T5 checkpoints
/// (pre-norm RMS, bucketed relative position bias) and the BART family (BART, mBART: learned positions, LayerNorm, post- or pre-norm).
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
/// <c>decoder_start_token</c> (default 0), <c>context_length</c> (default 512), <c>pre_norm</c> (default true: x + f(norm(x)); false:
/// norm(x + f(x)), BART), <c>embedding_scale</c> (factor of the token embeddings, default 1; mBART: √hidden_size).
/// <para>Weights: <c>token_embedding.weight</c> [V, H] (shared by encoder and decoder), optional <c>output.weight</c> [V, H] (default: the
/// embedding); per encoder layer <c>encoder.layers.{i}.</c> and decoder layer <c>decoder.layers.{i}.</c>: <c>attention_norm.weight</c>,
/// <c>attention.{q,k,v,o}.weight</c>, <c>mlp_norm.weight</c>, <c>mlp.up.weight</c>, optional <c>mlp.gate.weight</c> (gated: act(gate) · up),
/// <c>mlp.down.weight</c>; per decoder layer also <c>cross_attention_norm.weight</c> and <c>cross_attention.{q,k,v,o}.weight</c>. Every
/// projection and norm may have a <c>.bias</c>. <c>encoder.attention.relative_bias.weight</c> / <c>decoder.attention.relative_bias.weight</c>
/// [buckets, heads] (f32, f16 or bf16): T5's bucketed relative position bias, shared by all layers of the stack (a layer's own
/// <c>{stack}.layers.{i}.attention.relative_bias.weight</c> takes precedence; absent: no position information);
/// optional <c>encoder.output_norm.weight</c>, <c>decoder.output_norm.weight</c> (final norms); optional per stack
/// <c>{stack}.position_embedding.weight</c> [P, H] (learned absolute positions, row p is position p: importers remove offsets such as
/// BART's 2) and <c>{stack}.embedding_norm.weight</c> (norm of the summed embeddings); optional <c>output.bias</c> [V].</para>
/// <para>Speech encoders (<c>encoder.input</c> "audio", Whisper): the <see cref="WellKnown.EncodeGraph"/> takes one window of log-mel
/// features [<c>audio.mel_bins</c>, <c>audio.chunk_samples</c> / <c>audio.hop_length</c>] (<see cref="InputRole.AudioFeatures"/>, computed
/// on the host by the model's feature extractor) instead of tokens and runs the <see cref="AudioConvStem"/>. There is no embedding graph,
/// and the definition describes a <see cref="ModelKinds.SpeechRecognition"/> model (<see cref="SpeechRecognitionDescription"/>: encoder,
/// decoder, tokenizer and feature-extractor components, the filter bank and tokenizer as attachments).</para>
/// <para>Conformer encoders (<c>encoder.type</c> "conformer", Canary): the <see cref="WellKnown.EncodeGraph"/> is the
/// <see cref="ConformerEncoder"/> over a variable-length window of time-major log-mel features; the transformer encoder keys and weights
/// do not apply. An encoder wider than the decoder (Cohere Transcribe) has <c>encoder.hidden_size</c> and the optional
/// <c>encoder.projection.weight</c> [hidden_size, encoder.hidden_size] (with <c>.bias</c>), which projects the encoder output to the decoder
/// width.</para>
/// <para>Variable-length audio (<c>audio.variable_length</c>, with conformer encoders always): the encoder takes time-major features
/// [<see cref="WellKnown.AudioFrameSymbol"/>, <c>audio.channels</c> (default <c>audio.mel_bins</c>)] of up to <c>audio.max_frames</c>
/// frames, the cross states hold the rows of the longest window, and the decode graph's <see cref="InputRole.RowSpans"/> input
/// ([T, 2], each row the span (0, rows of the current window)) limits the cross-attention to the current window.</para>
/// <para>Rotary attention (<c>attention.rope.dims</c> &gt; 0, with <c>attention.rope.base</c> and <c>attention.rope.style</c>, Moonshine):
/// queries and keys of the audio encoder's and the decoder's self-attention are rotated by their row positions
/// (<see cref="InputRole.Positions"/> inputs of both graphs). Per-stack activations: <c>encoder.mlp.activation</c>,
/// <c>decoder.mlp.activation</c> (default <c>mlp.activation</c>). LayerNorms may lack biases.</para>
/// <para>Cached cross-attention (<c>cross_attention.cached</c>, for sources of a fixed length <c>encoder.rows</c>): the decoder projects the
/// cross-attention keys and values once per source into f32 states [rows, heads, head_dim] instead of on every step. The decode graph's
/// <see cref="InputRole.EncoderOutput"/> input then carries the source on the first pass of a context (E = rows) and no rows afterwards
/// (E = 0); the runtime scalar <see cref="WellKnown.SourcePastScalar"/> is 0.</para>
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
        string encoderAct = src.Arch("encoder.mlp.activation", act);
        string decoderAct = src.Arch("decoder.mlp.activation", act);
        long ropeDims = src.Arch("attention.rope.dims", 0L);
        float ropeBase = src.Arch("attention.rope.base", 10000f);
        string ropeStyle = src.Arch("attention.rope.style", "neox");
        float outputScale = src.Arch("output.scale", 1f);
        int ctx = src.Arch("context_length", 512);
        bool preNorm = src.Arch("pre_norm", true);
        float embeddingScale = src.Arch("embedding_scale", 1f);
        string encoderInput = src.Arch("encoder.input", "tokens");
        bool audio = encoderInput switch
        {
            "tokens" or "image" or "image_text" => false,
            "audio" => true,
            var other => throw new NotSupportedException($"encoder_decoder: unknown encoder.input '{other}'."),
        };
        bool image = encoderInput == "image";
        bool imageText = encoderInput == "image_text";
        bool encoderProjection = src.Weight("encoder.projection.weight") is not null;
        int crossWidth = encoderProjection ? nEmbd : src.Arch("encoder.hidden_size", nEmbd);
        bool cachedCross = src.Arch("cross_attention.cached", false);
        var extraWeights = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
        bool conformerEncoder = audio && src.Arch("encoder.type", "transformer") == "conformer";
        var conformer = conformerEncoder ? new ConformerEncoder(src, extraWeights) : null;
        var stem = audio && !conformerEncoder ? new AudioConvStem(src, extraWeights) : null;
        bool variableAudio = audio && (conformerEncoder || src.Arch("audio.variable_length", false));
        int audioChannels = audio ? src.Arch("audio.channels", src.Arch("audio.mel_bins", 1)) : 0;
        long audioFrames = audio ? (variableAudio ? src.Arch<long>("audio.max_frames") : src.Arch<int>("audio.chunk_samples") / src.Arch<int>("audio.hop_length")) : 0;
        Dim StemRows(Dim frames) => conformer?.Rows(frames) ?? stem!.Rows(frames);
        int sourceRows = !cachedCross ? 0 : variableAudio ? (int)StemRows(audioFrames).Fixed : src.Arch<int>("encoder.rows");
        WeightInfo? Lookup(string name) => extraWeights.TryGetValue(name, out var w) ? w : src.Weight(name);
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

        // Cross-attention keys / values of the current source, projected once per source (f32: the bidirectional attention kernels read f32 rows).
        var crossK = new StateDefinition[cachedCross ? decoderLayers : 0];
        var crossV = new StateDefinition[crossK.Length];
        for (int il = 0; il < crossK.Length; il++)
        {
            string g = $"decoder.layers.{il}";
            var shape = new Shape(sourceRows, nHead, headDim);
            states.Add(crossK[il] = new StateDefinition { Name = $"{g}.cross_k", Type = DataType.F32, Shape = shape, Kind = StateKind.Other, Group = g });
            states.Add(crossV[il] = new StateDefinition { Name = $"{g}.cross_v", Type = DataType.F32, Shape = shape, Kind = StateKind.Other, Group = g });
        }

        var graphs = new Dictionary<string, ComputeGraph>();
        if (!audio && !image && !imageText)
        {
            graphs[WellKnown.EmbedGraph] = BuildEmbed();
        }

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
        var speech = audio && decoderLayers > 0 ? SpeechRecognitionDescription.Build(src, ctx, sourceRows, variableAudio ? audioFrames : 0) : default((ModelDescription Description, IReadOnlyDictionary<string, IWeightSource> Attachments, IReadOnlyDictionary<string, object> Metadata)?);
        var recognition = (image || imageText) && decoderLayers > 0 ? TextRecognitionDescription.Build(src, ctx, sourceRows) : default((ModelDescription Description, IReadOnlyDictionary<string, object> Metadata, IReadOnlyDictionary<string, IWeightSource> Attachments)?);
        var weights = extraWeights.Count == 0 ? src.Weights : new Dictionary<string, WeightInfo>(src.Weights.Concat(extraWeights), StringComparer.Ordinal);

        return new ModelDefinition
        {
            Architecture = src.Architecture,
            Name = src.Get(CanonicalKeys.Name, src.Architecture),
            Weights = weights,
            Graphs = graphs,
            States = states,
            SourceMetadata = speech?.Metadata ?? recognition?.Metadata ?? src.Metadata,
            OwnedResources = src.OwnedResources,
            Properties = props,
            Description = speech?.Description ?? recognition?.Description,
            Attachments = speech?.Attachments ?? recognition?.Attachments ?? src.Attachments,
        };

        ComputeGraph BuildEncode()
        {
            var b = new GraphBuilder(WellKnown.EncodeGraph, Lookup);
            Value x;
            Value? encoderPositions = null;
            if (image)
            {
                var tower = new TowerOptions(src.Arch<int>("vision.layer_count"), src.Arch<int>("vision.hidden_size"), src.Arch<int>("vision.attention.head_count"),
                    src.Arch("vision.norm_eps", 1e-6f), src.Arch("vision.mlp.activation", "gelu"), "vision.");
                var towerImage = new ImageTowerOptions(tower, src.Arch<int>("vision.patch_size"), src.Arch<int>("vision.image_size"), src.Arch("vision.register_count", 0), "cls");
                int grid = towerImage.ImageSize / towerImage.Patch;
                var patches = b.Input("patches", InputRole.Media, DataType.F32, new Shape(grid * grid, 3 * towerImage.Patch * towerImage.Patch));
                var embedded = ImageTowerGraph.Embed(b, patches, towerImage, out int imageRows);
                embedded = TowerBlocks.Layers(b, embedded, tower, imageRows, causal: false);
                using (b.Group("output"))
                {
                    b.Output(WellKnown.EncoderOutput, TowerBlocks.LayerNorm(b, embedded, "vision.output_norm", tower.Eps));
                }

                return b.Build();
            }

            if (imageText)
            {
                return BuildImageTextEncode(b);
            }

            if (conformer is not null)
            {
                var encoded = conformer.Build(b);
                using (b.Group("output"))
                {
                    b.Output(WellKnown.EncoderOutput, encoderProjection ? Linear(b, encoded, "encoder.projection", "encoder_projection") : encoded);
                }

                return b.Build();
            }

            if (stem is not null)
            {
                // Fixed windows (Whisper) come channel-major [channels, frames]; variable-length input comes time-major [F, channels].
                Value signal;
                using (b.Group("input"))
                {
                    signal = variableAudio
                        ? b.Input("features", InputRole.AudioFeatures, DataType.F32, new Shape(Dim.Sym(WellKnown.AudioFrameSymbol), audioChannels))
                        : b.Permute(b.Input("features", InputRole.AudioFeatures, DataType.F32, new Shape(audioChannels, audioFrames)), 1, 0);
                }

                x = stem.Build(b, signal, (t, activation) => Act(b, t, activation));
                if (ropeDims > 0)
                {
                    using var _ = b.Group("input");
                    encoderPositions = b.Input("positions", InputRole.Positions, DataType.I32, new Shape(x.Shape[0], 1));
                }
            }
            else
            {
                using var _ = b.Group("input");
                var T = Dim.Sym(WellKnown.TokenSymbol);
                var tokens = b.Input("tokens", InputRole.TokenIds, DataType.I32, new Shape(T));
                var positions = HasPositions("encoder") ? b.Input("positions", InputRole.Positions, DataType.I32, new Shape(T, 1)) : null;
                x = Embeddings(b, "encoder", tokens, positions);
            }

            x = EncoderLayers(b, x, encoderPositions, (q, k, v, il, name) => b.Emit1("Attention", [q, k, v], Attributes.From([("scale", scale), ("causal", false), .. BiasAttributes(encoderBias[il])]), name));
            using (b.Group("output"))
            {
                b.Output(WellKnown.EncoderOutput, FinalNorm(b, x, "encoder.output_norm", "encoder_output"));
            }

            return b.Build();
        }

        // Image and prompt in one encoder (Florence-2): the DaViT tower's rows come first, then the prompt's token embeddings; learned positions over all rows, the
        // embedding norm and the bidirectional layers. The source length varies with the prompt: the decoder's cross-attention limits itself to the rows of the current source.
        ComputeGraph BuildImageTextEncode(GraphBuilder b)
        {
            var tower = new DaViTGraph(src, b, new ConstantCache(b));
            int imageSize = src.Arch<int>("vision.image_size");
            int imageRows = tower.Rows(imageSize);
            var pixels = b.Input("pixels", InputRole.Media, DataType.F32, new Shape(1, 3, imageSize, imageSize));
            var imageFeatures = tower.Build(pixels);
            b.Output(WellKnown.ImageFeaturesOutput, imageFeatures);
            Value x;
            using (b.Group("input"))
            {
                var T = Dim.Sym(WellKnown.TokenSymbol);
                var tokens = b.Input("tokens", InputRole.TokenIds, DataType.I32, new Shape(T));
                var positions = b.Input("positions", InputRole.Positions, DataType.I32, new Shape(Dim.Of(imageRows) + T, 1));
                x = b.Concat(0, imageFeatures, b.Gather(b.Weight("token_embedding.weight"), tokens, "inp_embd"));
                x = b.Add(x, b.Gather(b.Weight("encoder.position_embedding.weight"), b.Reshape(positions, 0), "pos_embd"));
                x = b.HasWeight("encoder.embedding_norm.weight") ? Norm(b, x, "encoder.embedding_norm") : x;
            }

            x = EncoderLayers(b, x, null, (q, k, v, il, name) => b.Emit1("Attention", [q, k, v], Attributes.From([("scale", scale), ("causal", false), .. BiasAttributes(encoderBias[il])]), name));
            using (b.Group("output"))
            {
                b.Output(WellKnown.EncoderOutput, FinalNorm(b, x, "encoder.output_norm", "encoder_output"));
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
                x = Embeddings(b, "encoder", inp.Tokens, HasPositions("encoder") ? inp.Positions : null);
            }

            x = EncoderLayers(b, x, ropeDims > 0 ? inp.Positions : null, (q, k, v, il, name) => PackedSequences.Attention(b, q, k, v, inp.RowSpans, scale, causal: false, name: name, positionBias: encoderBias[il]));
            using (b.Group("output"))
            {
                PackedSequences.EmbeddingsOutput(b, FinalNorm(b, x, "encoder.output_norm", "encoder_output"), inp.SequenceSpans, pooling, normalize);
            }

            return b.Build();
        }

        Value EncoderLayers(GraphBuilder b, Value x, Value? positions, Func<Value, Value, Value, int, string, Value> attention)
        {
            for (int il = 0; il < encoderLayers; il++)
            {
                using var _ = b.Group($"encoder.layers.{il}");
                string p = $"encoder.layers.{il}.";
                x = SubLayer(b, x, p + "attention_norm", h =>
                {
                    var q = Rotate(b, Heads(b, Linear(b, h, p + "attention.q")), positions);
                    var k = Rotate(b, Heads(b, Linear(b, h, p + "attention.k")), positions);
                    var attn = attention(q, k, Heads(b, Linear(b, h, p + "attention.v")), il, $"{p}attn");
                    return Linear(b, Merge(b, attn), p + "attention.o", $"{p}attn_out");
                });
                x = SubLayer(b, x, p + "mlp_norm", h => FeedForward(b, h, p, encoderAct));
            }

            return x;
        }

        ComputeGraph BuildDecode()
        {
            var b = new GraphBuilder(WellKnown.DecodeGraph, Lookup);
            var T = Dim.Sym(WellKnown.TokenSymbol);
            Value x;
            Value encoded;
            Value? positions;
            Value? sourceSpans = null;
            using (b.Group("input"))
            {
                var tokens = b.Input("tokens", InputRole.TokenIds, DataType.I32, new Shape(T));
                positions = HasPositions("decoder") || ropeDims > 0 ? b.Input("positions", InputRole.Positions, DataType.I32, new Shape(T, 1)) : null;
                encoded = b.Input(WellKnown.EncoderOutput, InputRole.EncoderOutput, DataType.F32, new Shape(Dim.Sym(WellKnown.EncoderSymbol), crossWidth));
                if ((variableAudio || imageText) && cachedCross)
                {
                    // The cross states hold up to sourceRows rows; each query row sees the rows of the current source only.
                    sourceSpans = b.Input("source_spans", InputRole.RowSpans, DataType.I32, new Shape(T, 2));
                }

                x = Embeddings(b, "decoder", tokens, HasPositions("decoder") ? positions : null);
            }

            var outRows = b.Input("output_rows", InputRole.OutputRows, DataType.I32, new Shape(Dim.Sym("R")));
            for (int il = 0; il < decoderLayers; il++)
            {
                using var _ = b.Group($"decoder.layers.{il}");
                string p = $"decoder.layers.{il}.";
                int layer = il;
                x = SubLayer(b, x, p + "attention_norm", h =>
                {
                    var kc = b.State(kCache[layer]);
                    var vc = b.State(vCache[layer]);
                    b.CacheWrite(kc, Rotate(b, Heads(b, Linear(b, h, p + "attention.k")), positions));
                    b.CacheWrite(vc, Heads(b, Linear(b, h, p + "attention.v")));
                    var attn = b.Emit1("CachedAttention", [Rotate(b, Heads(b, Linear(b, h, p + "attention.q")), positions), kc, vc],
                        Attributes.From([("scale", scale), ("offset", WellKnown.PastScalar), .. BiasAttributes(decoderBias[layer])]), $"{p}attn");
                    return Linear(b, Merge(b, attn), p + "attention.o", $"{p}attn_out");
                });

                // Cross-attention: keys and values projected from the encoder output, either on every step (short text sources) or once per
                // source into the cross states (written while the pass carries the source rows, a no-op for passes with E = 0).
                x = SubLayer(b, x, p + "cross_attention_norm", h =>
                {
                    Value keys = Heads(b, Linear(b, encoded, p + "cross_attention.k"));
                    Value values = Heads(b, Linear(b, encoded, p + "cross_attention.v"));
                    if (cachedCross)
                    {
                        var kc = b.State(crossK[layer]);
                        var vc = b.State(crossV[layer]);
                        b.CacheWrite(kc, keys, WellKnown.SourcePastScalar);
                        b.CacheWrite(vc, values, WellKnown.SourcePastScalar);
                        keys = kc;
                        values = vc;
                    }

                    var crossQuery = Heads(b, Linear(b, h, p + "cross_attention.q"));
                    var cross = b.Emit1("Attention", sourceSpans is null ? [crossQuery, keys, values] : [crossQuery, keys, values, sourceSpans],
                        Attributes.From(("scale", scale), ("causal", false)), $"{p}cross_attn");
                    return Linear(b, Merge(b, cross), p + "cross_attention.o", $"{p}cross_attn_out");
                });
                x = SubLayer(b, x, p + "mlp_norm", h => FeedForward(b, h, p, decoderAct));
            }

            using (b.Group("output"))
            {
                var hidden = FinalNorm(b, x, "decoder.output_norm", "result_norm");
                var rows = b.Gather(hidden, outRows, "selected_rows");
                if (outputScale != 1f)
                {
                    rows = b.Scale(rows, outputScale);
                }

                string head = b.HasWeight("output.weight") ? "output.weight" : "token_embedding.weight";
                b.Output(WellKnown.LogitsOutput, b.Linear(rows, head, b.HasWeight("output.bias") ? "output.bias" : null, "logits"));
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

        bool HasPositions(string stack) => src.Weight($"{stack}.position_embedding.weight") is not null;

        // Token embeddings (scaled), plus learned positions, then the optional embedding norm.
        Value Embeddings(GraphBuilder b, string stack, Value tokens, Value? positions)
        {
            var x = b.Gather(b.Weight("token_embedding.weight"), tokens, "inp_embd");
            if (embeddingScale != 1f)
            {
                x = b.Scale(x, embeddingScale);
            }

            if (positions is not null)
            {
                x = b.Add(x, b.Gather(b.Weight($"{stack}.position_embedding.weight"), b.Reshape(positions, 0), "pos_embd"));
            }

            return b.HasWeight($"{stack}.embedding_norm.weight") ? Norm(b, x, $"{stack}.embedding_norm") : x;
        }

        // A residual sub-layer: x + f(norm(x)) (pre-norm) or norm(x + f(x)) (post-norm).
        Value SubLayer(GraphBuilder b, Value x, string normName, Func<Value, Value> f)
            => preNorm ? b.Add(x, f(Norm(b, x, normName))) : Norm(b, b.Add(x, f(x)), normName);

        Value FinalNorm(GraphBuilder b, Value x, string normName, string name) => b.HasWeight(normName + ".weight") ? Norm(b, x, normName, name) : x;

        // Rows keep their (symbolic or, for fixed-size speech encoders, fixed) leading dimension.
        Value Heads(GraphBuilder b, Value v) => b.Reshape(v, Rows(v), nHead, headDim);

        Value Merge(GraphBuilder b, Value v) => b.Reshape(v, Rows(v), inner);

        static long Rows(Value v) => v.Shape[0].IsSymbolic ? 0 : v.Shape[0].Fixed;

        Value Linear(GraphBuilder b, Value v, string module, string? name = null)
            => b.Linear(v, module + ".weight", b.HasWeight(module + ".bias") ? module + ".bias" : null, name);

        // Rotary positions on the first attention.rope.dims of every head (self-attention only); identity without rope.
        Value Rotate(GraphBuilder b, Value heads, Value? positions)
            => ropeDims > 0 && positions is not null
                ? b.Emit1("Rope", [heads, positions], Attributes.From(("dims", ropeDims), ("base", ropeBase), ("style", ropeStyle)))
                : heads;

        Value Norm(GraphBuilder b, Value v, string module, string? name = null) => norm switch
        {
            "rms" => b.RmsNorm(v, b.Weight(module + ".weight"), eps, name),
            "layer" => b.Emit1("LayerNorm", b.HasWeight(module + ".bias") ? [v, b.Weight(module + ".weight"), b.Weight(module + ".bias")] : [v, b.Weight(module + ".weight")],
                Attributes.From(("eps", eps)), name),
            _ => throw new NotSupportedException($"encoder_decoder: unknown norm '{norm}'."),
        };

        Value FeedForward(GraphBuilder b, Value h, string p, string activation)
        {
            var up = Linear(b, h, p + "mlp.up");
            var y = b.HasWeight(p + "mlp.gate.weight") ? b.Mul(Act(b, Linear(b, h, p + "mlp.gate"), activation), up) : Act(b, up, activation);
            return Linear(b, y, p + "mlp.down", $"{p}ffn_out");
        }

        Value Act(GraphBuilder b, Value t, string? activation = null) => (activation ?? act) switch
        {
            "relu" => b.Unary("Relu", t),
            "gelu" => b.Unary("Gelu", t),
            "gelu_tanh" => b.Unary("GeluTanh", t),
            "silu" => b.Silu(t),
            "tanh" => b.Unary("Tanh", t),
            var other => throw new NotSupportedException($"encoder_decoder: unknown activation '{other}'."),
        };
    }
}
