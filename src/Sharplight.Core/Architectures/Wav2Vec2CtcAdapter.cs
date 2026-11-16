using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// wav2vec 2.0 with a CTC head (canonical arch <c>wav2vec2_ctc</c>): the speech recognizers of facebook/wav2vec2-base-960h and the XLSR-53
/// fine-tunes. One <see cref="WellKnown.EncodeGraph"/> over the raw waveform ([<see cref="WellKnown.AudioFrameSymbol"/>, 1], zero mean and
/// unit variance per recording when the feature extractor normalizes) that outputs the hidden rows and the CTC logits; the greedy CTC
/// decoding runs on the host like Parakeet's (<see cref="ConformerAsrAdapter"/>).
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>wav2vec2_ctc.</c>): <c>hidden_size</c>, <c>encoder.layer_count</c>, <c>encoder.attention.head_count</c>,
/// <c>encoder.mlp.activation</c> (default "gelu"), <c>encoder.norm_eps</c>, <c>encoder.stable_layer_norm</c> (pre-norm layers with the encoder
/// norm last; otherwise post-norm layers with the encoder norm right after the positional convolution), <c>encoder.conv.strides</c> (one per
/// feature convolution, unpadded), <c>encoder.conv.activation</c>, <c>encoder.conv.norm</c> ("group": a per-channel norm over time after the
/// first convolution; "layer": a layer norm over the channels after every convolution, before the activation), <c>encoder.conv.norm_eps</c>,
/// <c>encoder.position_conv.groups</c>, <c>decoder.type</c> ("ctc") and <c>decoder.blank_id</c>, the audio keys of
/// <see cref="SpeechRecognitionDescription"/> (waveform).
/// <para>Weights: <c>encoder.conv.{i}.weight</c> [C, C', K] (+ optional <c>.bias</c>), <c>encoder.conv.{i}.norm.{weight,bias}</c>,
/// <c>encoder.projection_norm.*</c>, <c>encoder.projection.*</c>, <c>encoder.position_conv.{weight,bias}</c> [H, H / groups, K] (weight norm
/// folded), <c>encoder.norm.*</c>, per layer <c>encoder.layers.{i}.</c>: <c>attention.{q,k,v,o}.{weight,bias}</c>, <c>attention_norm.*</c>,
/// <c>mlp.{up,down}.{weight,bias}</c>, <c>mlp_norm.*</c>; <c>ctc.weight</c> [tokens, H] and <c>ctc.bias</c>.</para>
/// </remarks>
public sealed class Wav2Vec2CtcAdapter : IModelAdapter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "wav2vec2_ctc";

    /// <inheritdoc/>
    public string Name => "wav2vec 2.0 CTC speech recognizer";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => [Architecture];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        int hidden = src.Arch<int>("hidden_size");
        int layers = src.Arch<int>("encoder.layer_count");
        int heads = src.Arch<int>("encoder.attention.head_count");
        int headDim = hidden / heads;
        float eps = src.Arch("encoder.norm_eps", 1e-5f);
        float convEps = src.Arch("encoder.conv.norm_eps", 1e-5f);
        bool stable = src.Arch("encoder.stable_layer_norm", false);
        string convNorm = src.Arch("encoder.conv.norm", "group");
        string convActivation = src.Arch("encoder.conv.activation", "gelu");
        string activation = src.Arch("encoder.mlp.activation", "gelu");
        var strides = src.Arch<long[]>("encoder.conv.strides");
        long maxFrames = src.Arch<long>("audio.max_frames");
        long groups = src.Arch("encoder.position_conv.groups", 16);

        var extraWeights = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
        var kernels = new long[strides.Length];
        for (int i = 0; i < strides.Length; i++)
        {
            string name = $"encoder.conv.{i}.weight";
            var w = src.Weight(name) ?? throw new NotSupportedException($"{Architecture}: missing '{name}'.");
            kernels[i] = w.Shape[2].Fixed;
            extraWeights[$"encoder.conv.{i}.kernel"] = ToF32(w, $"encoder.conv.{i}.kernel");
        }

        var positionKernel = src.Weight("encoder.position_conv.weight") ?? throw new NotSupportedException($"{Architecture}: missing 'encoder.position_conv.weight'.");
        long positionWidth = positionKernel.Shape[2].Fixed;
        extraWeights["encoder.position_conv.kernel"] = ToF32(positionKernel, "encoder.position_conv.kernel");
        WeightInfo? Lookup(string name) => extraWeights.TryGetValue(name, out var w) ? w : src.Weight(name);

        Dim Rows(Dim frames)
        {
            for (int i = 0; i < strides.Length; i++)
            {
                frames = (frames - kernels[i]).FloorDiv(strides[i]) + 1;
            }

            return frames;
        }

        var graphs = new Dictionary<string, ComputeGraph> { [WellKnown.EncodeGraph] = BuildEncode() };
        long rowFrames = maxFrames / Rows(maxFrames).Fixed;
        var decoderConfig = new Dictionary<string, object>
        {
            ["decoding"] = "ctc",
            ["blank"] = src.Arch<int>("decoder.blank_id"),
            ["rowFrames"] = rowFrames,
        };
        var encoderComponent = new ModelComponent
        {
            Role = ModelComponentRoles.Encoder,
            Graphs = [WellKnown.EncodeGraph],
            WeightPrefixes = ["encoder.", "ctc."],
            SymbolBounds = new Dictionary<string, long>(StringComparer.Ordinal) { [WellKnown.AudioFrameSymbol] = maxFrames },
        };
        var decoderComponent = new ModelComponent
        {
            Role = ModelComponentRoles.Decoder,
            Graphs = [],
            WeightPrefixes = [],
            Config = SpeechRecognitionDescription.JsonConfig(decoderConfig),
        };
        var speech = SpeechRecognitionDescription.Build(src, encoderComponent, decoderComponent, maxFrames);
        var weights = new Dictionary<string, WeightInfo>(src.Weights.Concat(extraWeights), StringComparer.Ordinal);
        return new ModelDefinition
        {
            Architecture = src.Architecture,
            Name = src.Get(CanonicalKeys.Name, src.Architecture),
            Weights = weights,
            Graphs = graphs,
            States = [],
            SourceMetadata = speech.Metadata,
            OwnedResources = src.OwnedResources,
            Properties = new Dictionary<string, object>
            {
                [WellKnown.VocabSize] = (int)src.Weight("ctc.weight")!.Shape[0].Fixed,
                [WellKnown.EmbeddingLength] = hidden,
                ["n_layer"] = layers,
            },
            Description = speech.Description,
            Attachments = speech.Attachments,
        };

        ComputeGraph BuildEncode()
        {
            var b = new GraphBuilder(WellKnown.EncodeGraph, Lookup);
            Value x;
            using (b.Group("input"))
            {
                x = b.Input("features", InputRole.AudioFeatures, DataType.F32, new Shape(Dim.Sym(WellKnown.AudioFrameSymbol), 1));
            }

            using (b.Group("encoder.conv"))
            {
                for (int i = 0; i < strides.Length; i++)
                {
                    string p = $"encoder.conv.{i}.";
                    Value[] inputs = b.HasWeight(p + "bias") ? [x, b.Weight(p + "kernel"), b.Weight(p + "bias")] : [x, b.Weight(p + "kernel")];
                    x = b.Emit1("Conv1d", inputs, Attributes.From(("stride", strides[i])), $"conv{i}");
                    if (b.HasWeight(p + "norm.weight"))
                    {
                        x = convNorm == "layer" ? LayerNorm(b, x, p + "norm", convEps) : ChannelNorm(b, x, p + "norm", convEps);
                    }

                    x = Activate(b, x, convActivation);
                }
            }

            using (b.Group("encoder.projection"))
            {
                x = LayerNorm(b, x, "encoder.projection_norm", convEps);
                x = b.Linear(x, "encoder.projection.weight", "encoder.projection.bias", "projection");
            }

            using (b.Group("encoder.position"))
            {
                long pad = positionWidth / 2;
                long padRight = positionWidth % 2 == 0 ? pad - 1 : pad;
                var conv = b.Emit1("Conv1d", [x, b.Weight("encoder.position_conv.kernel"), b.Weight("encoder.position_conv.bias")],
                    Attributes.From(("pad_left", pad), ("pad_right", padRight), ("groups", groups)), "position_conv");
                x = b.Add(x, Activate(b, conv, "gelu"));
                if (!stable)
                {
                    x = LayerNorm(b, x, "encoder.norm", eps);
                }
            }

            float scale = 1f / MathF.Sqrt(headDim);
            for (int il = 0; il < layers; il++)
            {
                using var _ = b.Group($"encoder.layers.{il}");
                string p = $"encoder.layers.{il}.";
                var h = stable ? LayerNorm(b, x, p + "attention_norm", eps) : x;
                var q = b.Reshape(b.Linear(h, p + "attention.q.weight", p + "attention.q.bias"), 0, heads, headDim);
                var k = b.Reshape(b.Linear(h, p + "attention.k.weight", p + "attention.k.bias"), 0, heads, headDim);
                var v = b.Reshape(b.Linear(h, p + "attention.v.weight", p + "attention.v.bias"), 0, heads, headDim);
                var attention = b.Emit1("Attention", [q, k, v], Attributes.From(("scale", scale), ("causal", false)), $"{p}attn");
                var o = b.Linear(b.Reshape(attention, 0, hidden), p + "attention.o.weight", p + "attention.o.bias", $"{p}attn_out");
                x = stable ? b.Add(x, o) : LayerNorm(b, b.Add(x, o), p + "attention_norm", eps);
                var m = stable ? LayerNorm(b, x, p + "mlp_norm", eps) : x;
                var ffn = b.Linear(Activate(b, b.Linear(m, p + "mlp.up.weight", p + "mlp.up.bias"), activation), p + "mlp.down.weight", p + "mlp.down.bias", $"{p}ffn_out");
                x = stable ? b.Add(x, ffn) : LayerNorm(b, b.Add(x, ffn), p + "mlp_norm", eps);
            }

            if (stable)
            {
                x = LayerNorm(b, x, "encoder.norm", eps);
            }

            using (b.Group("output"))
            {
                b.Output(WellKnown.HiddenOutput, x);
                b.Output(WellKnown.LogitsOutput, b.Linear(x, "ctc.weight", "ctc.bias", "logits"));
            }

            return b.Build();
        }
    }

    private static WeightInfo ToF32(WeightInfo w, string name) => w.Type == DataType.F32 ? w.As(name, w.Shape) : WeightConversions.ToF32(w, name, w.Shape);

    private static Value LayerNorm(GraphBuilder b, Value x, string module, float eps)
        => b.Emit1("LayerNorm", [x, b.Weight(module + ".weight"), b.Weight(module + ".bias")], Attributes.From(("eps", eps)));

    // A group norm with one group per channel over the time-major signal [T, C]: mean and variance over time, then a scale and shift per channel.
    private static Value ChannelNorm(GraphBuilder b, Value x, string module, float eps)
    {
        var time = Attributes.From(("mode", "mean"), ("axes", new long[] { 0 }), ("keepdims", true));
        var centered = b.Sub(x, b.Emit1("Reduce", [x], time));
        var variance = b.Emit1("Reduce", [b.Unary("Square", centered)], time);
        var normed = b.Mul(centered, b.Unary("Rsqrt", b.Scale(variance, 1f, eps)));
        return b.Add(b.Mul(normed, b.Weight(module + ".weight")), b.Weight(module + ".bias"), module);
    }

    private static Value Activate(GraphBuilder b, Value x, string activation) => activation switch
    {
        "gelu" => b.Unary("Gelu", x),
        "gelu_new" or "gelu_tanh" => b.Unary("GeluTanh", x),
        "relu" => b.Unary("Relu", x),
        "silu" => b.Silu(x),
        _ => throw new NotSupportedException($"{Architecture}: unknown activation '{activation}'."),
    };
}
