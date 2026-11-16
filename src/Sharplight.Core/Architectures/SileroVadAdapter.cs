using Sharplight.Audio;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// The Silero voice activity detector (canonical arch <c>silero_vad</c>, kind <see cref="ModelKinds.VoiceActivity"/>): a learned magnitude STFT as a strided convolution, four
/// convolution blocks, one LSTM step carrying the state across frames and a sigmoid head. Builds the stateful per-frame graph <see cref="VoiceActivityGraphs.StepGraph"/>.
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>silero_vad.*</c>): <c>sample_rate</c>, <c>frame_samples</c>, <c>context_samples</c>, <c>hop</c> (the STFT hop), <c>hidden_size</c>, <c>pad</c> (the reflect padding
/// at the end of a frame), <c>strides</c> (long[] of the encoder convolutions). Weights: <c>stft.basis</c> [2·bins, 1, n_fft], <c>encoder.{i}.weight</c> and <c>.bias</c>,
/// <c>rnn.{weight_ih,weight_hh,bias_ih,bias_hh}</c>, <c>head.weight</c> and <c>head.bias</c>.
/// </remarks>
public sealed class SileroVadAdapter : IModelAdapter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "silero_vad";

    /// <inheritdoc/>
    public string Name => "Silero VAD";

    /// <inheritdoc/>
    public IReadOnlyList<string> Architectures => [Architecture];

    /// <inheritdoc/>
    public ModelDefinition Build(ModelSourceSet sources, AdapterOptions options)
    {
        var src = sources.Primary;
        int sampleRate = src.Arch<int>("sample_rate");
        int frame = src.Arch<int>("frame_samples");
        int context = src.Arch<int>("context_samples");
        int hidden = src.Arch<int>("hidden_size");
        long hop = src.Arch<int>("hop");
        long pad = src.Arch<int>("pad");
        long[] strides = src.Arch<long[]>("strides");

        var b = new GraphBuilder(VoiceActivityGraphs.StepGraph, src.Weight);
        var audio = b.Input(VoiceActivityGraphs.AudioInput, InputRole.Media, DataType.F32, new Shape(context + frame, 1));
        var state = b.Input(VoiceActivityGraphs.StateInput, InputRole.Other, DataType.F32, new Shape(2, hidden));
        var spectrum = b.Emit1("Conv1d", [audio, b.Weight("stft.basis")], Attributes.From(("stride", hop), ("pad_left", 0L), ("pad_right", pad), ("pad_mode", "reflect")));
        long bins = spectrum.Shape[1].Fixed / 2;
        var x = b.Unary("Sqrt", b.Add(b.Unary("Square", b.Slice(spectrum, 1, 0, bins)), b.Unary("Square", b.Slice(spectrum, 1, bins, bins))));
        for (int i = 0; i < strides.Length; i++)
        {
            x = b.Emit1("Conv1d", [x, b.Weight($"encoder.{i}.weight"), b.Weight($"encoder.{i}.bias")],
                Attributes.From(("stride", strides[i]), ("pad_left", 1L), ("pad_right", 1L)));
            x = b.Unary("Relu", x);
        }

        var h = b.Slice(state, 0, 0, 1);
        var c = b.Slice(state, 0, 1, 1);
        var gates = b.Add(b.Add(b.MatMul(x, b.Weight("rnn.weight_ih")), b.Weight("rnn.bias_ih")), b.Add(b.MatMul(h, b.Weight("rnn.weight_hh")), b.Weight("rnn.bias_hh")));
        var input = b.Unary("Sigmoid", b.Slice(gates, 1, 0, hidden));
        var forget = b.Unary("Sigmoid", b.Slice(gates, 1, hidden, hidden));
        var cell = b.Unary("Tanh", b.Slice(gates, 1, 2L * hidden, hidden));
        var output = b.Unary("Sigmoid", b.Slice(gates, 1, 3L * hidden, hidden));
        var nextCell = b.Add(b.Mul(forget, c), b.Mul(input, cell));
        var nextHidden = b.Mul(output, b.Unary("Tanh", nextCell));
        b.Output(VoiceActivityGraphs.StateOutput, b.Concat(0, nextHidden, nextCell));
        var logit = b.Emit1("Conv1d", [b.Unary("Relu", nextHidden), b.Weight("head.weight"), b.Weight("head.bias")]);
        b.Output(VoiceActivityGraphs.ProbabilityOutput, b.Unary("Sigmoid", logit));

        var description = new ModelDescription
        {
            Kind = ModelKinds.VoiceActivity,
            Components =
            [
                new ModelComponent
                {
                    Role = ModelComponentRoles.Encoder,
                    Graphs = [VoiceActivityGraphs.StepGraph],
                    Config = ComponentConfig.Of((VoiceActivityGraphs.SampleRateKey, sampleRate), (VoiceActivityGraphs.FrameSamplesKey, frame),
                        (VoiceActivityGraphs.ContextSamplesKey, context)),
                },
            ],
        };
        return new ModelDefinition
        {
            Architecture = src.Architecture,
            Name = src.Get(CanonicalKeys.Name, src.Architecture),
            Weights = src.Weights,
            Graphs = new Dictionary<string, ComputeGraph>(StringComparer.Ordinal) { [VoiceActivityGraphs.StepGraph] = b.Build() },
            SourceMetadata = src.Metadata,
            OwnedResources = src.OwnedResources,
            Description = description,
        };
    }
}
