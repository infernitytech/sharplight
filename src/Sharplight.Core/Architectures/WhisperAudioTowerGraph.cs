using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Emits the audio tower graph (<see cref="ChunkedAudioTowerGraph.AudioEncodeGraph"/>) of a Whisper-encoder audio language model (Voxtral):
/// one fixed window of log-mel frames through Whisper's two <c>Conv1d</c> layers (strides 1 and 2, GELU) and learned positions, pre-norm
/// transformer layers over the whole window, the output LayerNorm, then every <see cref="AudioTowerParameters.Downsample"/> consecutive rows
/// concatenated into one and the two-layer GELU projector into the language model's width.
/// </summary>
internal static class WhisperAudioTowerGraph
{
    /// <summary>The name of the f32 kernel view of convolution <paramref name="index"/>.</summary>
    /// <param name="index">The convolution (0-based).</param>
    public static string KernelName(int index) => $"{AudioTowerParameters.WeightPrefix}conv.{index}.kernel";

    /// <summary>The f32 [O, C, K] kernel views of the convolutions (the generic <c>Conv1d</c> takes f32 kernels).</summary>
    /// <param name="p">The tower parameters.</param>
    /// <param name="weight">Looks up a weight.</param>
    /// <exception cref="NotSupportedException">A convolution weight is missing or not 3-D.</exception>
    public static IEnumerable<WeightInfo> DerivedWeights(AudioTowerParameters p, Func<string, WeightInfo?> weight)
    {
        for (int i = 0; i < p.ConvCount; i++)
        {
            string name = $"{AudioTowerParameters.WeightPrefix}conv.{i}.weight";
            var w = weight(name) ?? throw new NotSupportedException($"The audio tower lacks '{name}'.");
            if (w.Shape.Rank != 3)
            {
                throw new NotSupportedException($"'{name}' has shape {w.Shape}; an [out, in, kernel] kernel was expected.");
            }

            yield return w.Type == DataType.F32 ? w.As(KernelName(i), w.Shape) : WeightConversions.ToF32(w, KernelName(i), w.Shape);
        }
    }

    /// <summary>Builds the graph.</summary>
    /// <param name="p">The tower parameters.</param>
    /// <param name="weights">Looks up the canonical and derived weights.</param>
    public static ComputeGraph Build(AudioTowerParameters p, Func<string, WeightInfo?> weights)
    {
        const string P = AudioTowerParameters.WeightPrefix;
        var b = new GraphBuilder(ChunkedAudioTowerGraph.AudioEncodeGraph, weights);
        Value x;
        using (b.Group("audio.stem"))
        {
            x = b.Input(ChunkedAudioTowerGraph.FeaturesInput, InputRole.AudioFeatures, DataType.F32, new Shape(p.WindowFrames, p.MelBins));
            for (int i = 0; i < p.ConvCount; i++)
            {
                string bias = $"{P}conv.{i}.bias";
                Value[] inputs = b.HasWeight(bias) ? [x, b.Weight(KernelName(i)), b.Weight(bias)] : [x, b.Weight(KernelName(i))];
                var attributes = Attributes.From(("stride", i == 0 ? 1L : 2L), ("pad_left", 1L), ("pad_right", 1L));
                x = TowerBlocks.Activation(b, b.Emit1("Conv1d", inputs, attributes, $"audio.conv{i}"), "gelu");
            }

            x = b.Add(x, b.Weight($"{P}position_embedding.weight"), "audio.positioned");
        }

        x = AudioTowerLayers.Build(b, x, p, spans: null);
        x = TowerBlocks.LayerNorm(b, x, P + "output_norm", p.Eps);
        long rows = x.Shape[0].Fixed;
        x = b.Reshape(x, rows / p.Downsample, (long)p.Hidden * p.Downsample);
        x = AudioTowerLayers.Linear(b, x, P + "projector.1", "audio.projector1");
        x = TowerBlocks.Activation(b, x, p.Activation);
        x = AudioTowerLayers.Linear(b, x, P + "projector.2", ChunkedAudioTowerGraph.EmbeddingsOutput);
        b.Output(ChunkedAudioTowerGraph.EmbeddingsOutput, x);
        return b.Build();
    }
}
