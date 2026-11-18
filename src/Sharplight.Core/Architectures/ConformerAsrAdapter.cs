using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Speech recognizers with a FastConformer encoder and an alignment decoder (canonical arch <c>conformer_asr</c>: NVIDIA Parakeet CTC, RNN-T
/// and TDT). Builds <see cref="WellKnown.EncodeGraph"/> over a variable-length window of log-mel features (<see cref="ConformerEncoder"/>)
/// whose output is either the CTC logits [rows, tokens] (<see cref="WellKnown.LogitsOutput"/>) or, for transducers, the encoder rows
/// projected into the joint space [rows, J] (<see cref="WellKnown.EncoderOutput"/>), plus the encoder rows themselves
/// (<see cref="WellKnown.HiddenOutput"/>, skipped unless asked for); transducers also get the
/// <see cref="WellKnown.TransducerPredictGraph"/> (embedding and LSTM layers, the states kept by the host) and the
/// <see cref="WellKnown.TransducerJointGraph"/>. The greedy decoding loops run on the host (the CTC and transducer recognizers of the task
/// layer) and read the decoder component's configuration.
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>conformer_asr.</c>): the encoder's (<see cref="ConformerEncoder"/>) and the audio front end's
/// (<see cref="SpeechRecognitionDescription"/>; <c>audio.max_frames</c> bounds a window), <c>decoder.type</c> ("ctc", "rnnt" or "tdt"),
/// <c>decoder.blank_id</c>, transducers: <c>decoder.layer_count</c> (LSTM layers), optional <c>decoder.durations</c> (TDT: the frame advances
/// the joint's duration logits stand for), <c>decoder.max_symbols_per_step</c> (default 10), <c>joint.activation</c> (default "relu").
/// Streaming models (Nemotron ASR streaming, <see cref="ConformerEncoder"/>'s causal keys) also have <c>encoder.attention.left_context</c> and
/// <c>encoder.attention.lookahead</c> (rows; the attention chunks are <c>lookahead + 1</c> rows, each row sees its chunk and the chunks of the left context) and
/// an optional language prompt: <c>prompt.count</c> (one-hot size), <c>prompt.default_id</c>, <c>prompt.dictionary</c> ("name=index" pairs separated by commas) with
/// the weights <c>prompt.linear_1.hidden.weight</c> [P, H] with <c>.bias</c>, <c>prompt.linear_1.prompt.weight</c> [P, count] and <c>prompt.linear_2.weight</c> [H, P] with
/// <c>.bias</c>: the encoder rows pass through <c>linear_2(relu(linear_1([row, one-hot])))</c> before the joint projection.
/// <para>Weights: CTC <c>ctc.weight</c> [tokens, H] and <c>ctc.bias</c>; transducers <c>joint.encoder.weight</c> [J, H] and <c>.bias</c>,
/// <c>decoder.embedding.weight</c> [tokens, J], per LSTM layer <c>decoder.lstm.{l}.input.weight</c> and <c>decoder.lstm.{l}.hidden.weight</c>
/// [4J, J] (gates in the order input, forget, cell, output) with optional <c>.bias</c>, <c>decoder.projection.weight</c> [J, J] with optional
/// <c>.bias</c>, <c>joint.output.weight</c> [tokens + durations, J] and <c>.bias</c>.</para>
/// </remarks>
public sealed class ConformerAsrAdapter : IModelAdapter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "conformer_asr";

    /// <inheritdoc/>
    public string Name => "FastConformer speech recognizer";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => [Architecture];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        var extraWeights = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
        var encoder = new ConformerEncoder(src, extraWeights);
        WeightInfo? Lookup(string name) => extraWeights.TryGetValue(name, out var w) ? w : src.Weight(name);
        string decoding = src.Arch<string>("decoder.type");
        bool transducer = decoding switch
        {
            "ctc" => false,
            "rnnt" or "tdt" => true,
            var other => throw new NotSupportedException($"{Architecture}: unknown decoder.type '{other}'."),
        };
        bool hasPrompt = src.Weight("prompt.linear_2.weight") is not null;
        long maxFrames = src.Arch<long>("audio.max_frames");
        int lstmLayers = transducer ? src.Arch<int>("decoder.layer_count") : 0;
        int jointSize = transducer ? (int)src.Weight("joint.encoder.weight")!.Shape[0].Fixed : 0;
        var graphs = new Dictionary<string, ComputeGraph> { [WellKnown.EncodeGraph] = BuildEncode() };
        if (transducer)
        {
            graphs[WellKnown.TransducerPredictGraph] = BuildPredict();
            graphs[WellKnown.TransducerJointGraph] = BuildJoint();
        }

        long rowFrames = maxFrames / encoder.Rows(maxFrames).Fixed;
        var decoderConfig = new Dictionary<string, object>
        {
            ["decoding"] = decoding,
            ["blank"] = src.Arch<int>("decoder.blank_id"),
            ["rowFrames"] = rowFrames,
        };
        if (transducer)
        {
            decoderConfig["durations"] = src.Arch("decoder.durations", Array.Empty<long>());
            decoderConfig["maxSymbolsPerStep"] = src.Arch("decoder.max_symbols_per_step", 10);
            decoderConfig["layers"] = lstmLayers;
        }

        var encoderConfig = new Dictionary<string, object>();
        if (src.Arch("encoder.attention.chunked", false))
        {
            encoderConfig["chunkRows"] = src.Arch("encoder.attention.lookahead", 0) + 1;
            encoderConfig["leftContext"] = src.Arch("encoder.attention.left_context", 0);
        }

        if (hasPrompt)
        {
            encoderConfig["promptCount"] = src.Arch<int>("prompt.count");
            encoderConfig["promptDefault"] = src.Arch("prompt.default_id", 0);
            encoderConfig["promptDictionary"] = src.Arch("prompt.dictionary", "");
        }

        var encoderComponent = new ModelComponent
        {
            Role = ModelComponentRoles.Encoder,
            Config = SpeechRecognitionDescription.JsonConfig(encoderConfig),
            Graphs = [WellKnown.EncodeGraph],
            WeightPrefixes = ["encoder.", "ctc.", "joint.encoder.", "prompt."],
            SymbolBounds = new Dictionary<string, long>(StringComparer.Ordinal) { [WellKnown.AudioFrameSymbol] = maxFrames },
        };
        var decoderComponent = new ModelComponent
        {
            Role = ModelComponentRoles.Decoder,
            Graphs = transducer ? [WellKnown.TransducerPredictGraph, WellKnown.TransducerJointGraph] : [],
            WeightPrefixes = transducer ? ["decoder.", "joint.output."] : [],
            Config = SpeechRecognitionDescription.JsonConfig(decoderConfig),
        };
        var speech = SpeechRecognitionDescription.Build(src, encoderComponent, decoderComponent, maxFrames);
        var weights = new Dictionary<string, WeightInfo>(src.Weights.Concat(extraWeights), StringComparer.Ordinal);
        int vocabulary = (int)(transducer ? src.Weight("decoder.embedding.weight")! : src.Weight("ctc.weight")!).Shape[0].Fixed;
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
                [WellKnown.VocabSize] = vocabulary,
                [WellKnown.EmbeddingLength] = encoder.Hidden,
                ["n_layer"] = src.Arch<int>("encoder.layer_count"),
            },
            Description = speech.Description,
            Attachments = speech.Attachments,
        };

        ComputeGraph BuildEncode()
        {
            var b = new GraphBuilder(WellKnown.EncodeGraph, Lookup);
            var x = encoder.Build(b);
            var fused = x;
            if (hasPrompt)
            {
                using var _ = b.Group("prompt");
                var prompt = b.Input("prompt", InputRole.Other, DataType.F32, new Shape(1, src.Arch<int>("prompt.count")));
                var hiddenPart = Linear(b, x, "prompt.linear_1.hidden");
                var promptPart = b.Linear(prompt, "prompt.linear_1.prompt.weight", null);
                fused = Linear(b, b.Unary("Relu", b.Add(hiddenPart, promptPart)), "prompt.linear_2");
            }

            using (b.Group("output"))
            {
                b.Output(WellKnown.HiddenOutput, x);
                if (transducer)
                {
                    b.Output(WellKnown.EncoderOutput, Linear(b, fused, "joint.encoder", "encoder_output"));
                }
                else
                {
                    b.Output(WellKnown.LogitsOutput, Linear(b, x, "ctc", "logits"));
                }
            }

            return b.Build();
        }

        // One prediction step: the token's embedding through the LSTM layers (states in, advanced states out), then the projection.
        ComputeGraph BuildPredict()
        {
            var b = new GraphBuilder(WellKnown.TransducerPredictGraph, Lookup);
            Value x;
            Value hidden;
            Value cell;
            using (b.Group("input"))
            {
                var token = b.Input("tokens", InputRole.TokenIds, DataType.I32, new Shape(1));
                hidden = b.Input(WellKnown.TransducerHiddenState, InputRole.Other, DataType.F32, new Shape(lstmLayers, jointSize));
                cell = b.Input(WellKnown.TransducerCellState, InputRole.Other, DataType.F32, new Shape(lstmLayers, jointSize));
                x = b.Gather(b.Weight("decoder.embedding.weight"), token, "embedding");
            }

            var hiddenOut = new Value[lstmLayers];
            var cellOut = new Value[lstmLayers];
            for (int l = 0; l < lstmLayers; l++)
            {
                using var _ = b.Group($"decoder.lstm.{l}");
                string p = $"decoder.lstm.{l}.";
                var h = b.Slice(hidden, 0, l, 1);
                var c = b.Slice(cell, 0, l, 1);
                var gates = b.Add(Linear(b, x, p + "input"), Linear(b, h, p + "hidden"));
                var input = b.Unary("Sigmoid", b.Slice(gates, 1, 0, jointSize));
                var forget = b.Unary("Sigmoid", b.Slice(gates, 1, jointSize, jointSize));
                var candidate = b.Unary("Tanh", b.Slice(gates, 1, 2 * jointSize, jointSize));
                var output = b.Unary("Sigmoid", b.Slice(gates, 1, 3 * jointSize, jointSize));
                cellOut[l] = b.Add(b.Mul(forget, c), b.Mul(input, candidate));
                hiddenOut[l] = b.Mul(output, b.Unary("Tanh", cellOut[l]));
                x = hiddenOut[l];
            }

            using (b.Group("output"))
            {
                b.Output(WellKnown.TransducerPrediction, Linear(b, x, "decoder.projection", "prediction"));
                b.Output(WellKnown.TransducerHiddenState, lstmLayers == 1 ? hiddenOut[0] : b.Concat(0, hiddenOut));
                b.Output(WellKnown.TransducerCellState, lstmLayers == 1 ? cellOut[0] : b.Concat(0, cellOut));
            }

            return b.Build();
        }

        ComputeGraph BuildJoint()
        {
            var b = new GraphBuilder(WellKnown.TransducerJointGraph, Lookup);
            Value frame;
            Value prediction;
            using (b.Group("input"))
            {
                frame = b.Input(WellKnown.EncoderOutput, InputRole.EncoderOutput, DataType.F32, new Shape(1, jointSize));
                prediction = b.Input(WellKnown.TransducerPrediction, InputRole.HiddenStates, DataType.F32, new Shape(1, jointSize));
            }

            using (b.Group("output"))
            {
                var joint = b.Add(frame, prediction);
                joint = src.Arch("joint.activation", "relu") switch
                {
                    "relu" => b.Unary("Relu", joint),
                    "tanh" => b.Unary("Tanh", joint),
                    var other => throw new NotSupportedException($"{Architecture}: unknown joint.activation '{other}'."),
                };
                b.Output(WellKnown.LogitsOutput, Linear(b, joint, "joint.output", "logits"));
            }

            return b.Build();
        }

        static Value Linear(GraphBuilder b, Value v, string module, string? name = null)
            => b.Linear(v, module + ".weight", b.HasWeight(module + ".bias") ? module + ".bias" : null, name);
    }
}
