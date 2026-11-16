using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Chatterbox's voice encoder (<c>VoiceEncoder.forward</c>) for one partial utterance: the mel frames through three LSTM layers, the last hidden state
/// through the projection and a ReLU. The host L2-normalizes each partial's embedding, averages the partials and normalizes again
/// (<see cref="ChatterboxVoiceCloning"/>).
/// </summary>
internal static class ChatterboxVoiceEncoderGraph
{
    /// <summary>The graph name.</summary>
    public const string GraphName = "voice_encoder";

    /// <summary>The mel frames of one partial [frames, melBins].</summary>
    public const string MelInput = "mel";

    /// <summary>The unnormalized embedding [1, width].</summary>
    public const string EmbeddingOutput = "embedding";

    private const string P = "ve.";

    /// <summary>Builds the graph.</summary>
    /// <param name="weights">Looks up the weights.</param>
    /// <param name="frames">The frames of a partial.</param>
    /// <param name="melBins">The mel channels.</param>
    /// <param name="layers">The LSTM layers.</param>
    public static ComputeGraph Build(Func<string, WeightInfo?> weights, int frames, int melBins, int layers)
    {
        var b = new GraphBuilder(GraphName, weights);
        var h = b.Input(MelInput, InputRole.Other, DataType.F32, new Shape(frames, melBins));
        for (int l = 0; l < layers; l++)
        {
            // PyTorch keeps two biases per layer; both add to the input projection.
            var gates = b.Add(b.Linear(h, P + $"lstm.weight_ih_l{l}", P + $"lstm.bias_ih_l{l}"), b.Weight(P + $"lstm.bias_hh_l{l}"));
            h = b.Emit1("Lstm", [gates, b.Weight(P + $"lstm.weight_hh_l{l}")], Attributes.From(("reverse", false)));
        }

        var last = b.Slice(h, 0, frames - 1, 1);
        b.Output(EmbeddingOutput, b.Unary("Relu", b.Linear(last, P + "proj.weight", P + "proj.bias")));
        return b.Build();
    }
}
