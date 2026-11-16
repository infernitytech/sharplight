using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Emits the <see cref="AudioEncodeGraph"/> of a chunked audio tower (<see cref="AudioTowerParameters"/>) from generic ops: the strided
/// <c>SequenceConv2d</c> stem over the chunk-padded feature map (row masks between the convolutions keep the chunks independent, see
/// <see cref="Speech.ChunkedAudioLayout"/>), the projection of the flattened channels plus the in-chunk sinusoid positions, a <c>Gather</c> of the
/// valid rows, pre-norm transformer layers whose attention is limited to row spans (the windows), the output LayerNorm and the GELU projector.
/// </summary>
internal static class ChunkedAudioTowerGraph
{
    /// <summary>The graph name.</summary>
    public const string AudioEncodeGraph = "audio_encode";

    /// <summary>The time-major feature map [<see cref="WellKnown.AudioFrameSymbol"/>, mel bins].</summary>
    public const string FeaturesInput = "features";

    /// <summary>The in-chunk position of every row after the stem.</summary>
    public const string PositionsInput = "positions";

    /// <summary>The rows after the stem that hold valid frames [<see cref="WellKnown.EncoderSymbol"/>].</summary>
    public const string RowsInput = "valid_rows";

    /// <summary>The attention window (first row, count) of every valid row.</summary>
    public const string SpansInput = "window_spans";

    /// <summary>The output: one embedding row per valid row, in the language model's width.</summary>
    public const string EmbeddingsOutput = WellKnown.EmbeddingsOutput;

    /// <summary>The row mask input after convolution <paramref name="level"/> (1-based).</summary>
    /// <param name="level">The convolution.</param>
    public static string MaskInput(int level) => $"row_mask_{level}";

    /// <summary>The name of the derived sequence-convolution kernel of convolution <paramref name="index"/>.</summary>
    /// <param name="index">The convolution (0-based).</param>
    public static string KernelName(int index) => $"{AudioTowerParameters.WeightPrefix}conv.{index}.kernel";

    /// <summary>
    /// The derived weights the graph needs: each convolution's kernel with its two spatial axes swapped into the
    /// [O, C, frames, mel bins] order of <c>SequenceConv2d</c> (f32).
    /// </summary>
    /// <param name="p">The tower parameters.</param>
    /// <param name="weight">Looks up a weight.</param>
    /// <exception cref="NotSupportedException">A convolution weight is missing or not 4-D.</exception>
    public static IEnumerable<WeightInfo> DerivedWeights(AudioTowerParameters p, Func<string, WeightInfo?> weight)
    {
        for (int i = 0; i < p.ConvCount; i++)
        {
            string name = $"{AudioTowerParameters.WeightPrefix}conv.{i}.weight";
            var w = weight(name) ?? throw new NotSupportedException($"The audio tower lacks '{name}'.");
            if (w.Shape.Rank != 4)
            {
                throw new NotSupportedException($"'{name}' has shape {w.Shape}; a [out, in, mel, frames] kernel was expected.");
            }

            long o = w.Shape[0].Fixed, c = w.Shape[1].Fixed, kf = w.Shape[2].Fixed, kt = w.Shape[3].Fixed;
            yield return WeightInfo.Derived(KernelName(i), DataType.F32, new Shape(o, c, kt, kf), () =>
            {
                var source = WeightConversions.ReadFloats(w);
                var result = new float[source.Length];
                for (long oc = 0; oc < o * c; oc++)
                {
                    for (long f = 0; f < kf; f++)
                    {
                        for (long t = 0; t < kt; t++)
                        {
                            result[(oc * kt * kf) + (t * kf) + f] = source[(oc * kf * kt) + (f * kt) + t];
                        }
                    }
                }

                return MemoryMarshal.AsBytes(result.AsSpan()).ToArray();
            });
        }
    }

    /// <summary>Builds the graph.</summary>
    /// <param name="p">The tower parameters.</param>
    /// <param name="weights">Looks up the canonical and derived weights.</param>
    public static ComputeGraph Build(AudioTowerParameters p, Func<string, WeightInfo?> weights)
    {
        const string P = AudioTowerParameters.WeightPrefix;
        var b = new GraphBuilder(AudioEncodeGraph, weights);
        var rows = Dim.Sym(WellKnown.EncoderSymbol);
        Value x;
        Value spans;
        using (b.Group("audio.stem"))
        {
            var features = b.Input(FeaturesInput, InputRole.AudioFeatures, DataType.F32, new Shape(Dim.Sym(WellKnown.AudioFrameSymbol), p.MelBins));
            x = b.Reshape(features, 0, 1, p.MelBins);
            for (int i = 0; i < p.ConvCount; i++)
            {
                var attributes = Attributes.From(("strides", new long[] { 2, 2 }), ("pads", new long[] { 1, 1, 1, 1 }), ("groups", 1L));
                x = b.Emit1("SequenceConv2d", [x, b.Weight(KernelName(i)), b.Weight($"{P}conv.{i}.bias")], attributes, $"audio.conv{i}");
                x = TowerBlocks.Activation(b, x, "gelu");
                if (i + 1 < p.ConvCount)
                {
                    var mask = b.Input(MaskInput(i + 1), InputRole.Other, DataType.F32, new Shape(x.Shape[0], 1, 1));
                    x = b.Mul(x, mask, $"audio.conv{i}_masked");
                }
            }

            x = b.Reshape(x, 0, x.Shape[1].Fixed * x.Shape[2].Fixed);
            x = b.Linear(x, $"{P}conv_out.weight", null, "audio.conv_out");
            var positions = b.Input(PositionsInput, InputRole.Positions, DataType.I32, new Shape(x.Shape[0]));
            x = b.Add(x, b.Gather(PositionTable(b, p), positions), "audio.positioned");
            var valid = b.Input(RowsInput, InputRole.Other, DataType.I32, new Shape(rows));
            x = b.Gather(x, valid, "audio.valid_rows");
            spans = b.Input(SpansInput, InputRole.RowSpans, DataType.I32, new Shape(rows, 2));
        }

        x = AudioTowerLayers.Build(b, x, p, spans);
        x = TowerBlocks.LayerNorm(b, x, P + "output_norm", p.Eps);
        x = b.Linear(x, P + "projector.1.weight", P + "projector.1.bias", "audio.projector1");
        x = TowerBlocks.Activation(b, x, p.Activation);
        x = b.Linear(x, P + "projector.2.weight", P + "projector.2.bias", EmbeddingsOutput);
        b.Output(EmbeddingsOutput, x);
        return b.Build();
    }

    // Whisper's sinusoids (sines of all frequencies, then cosines) for the rows of one chunk.
    private static Value PositionTable(GraphBuilder b, AudioTowerParameters p)
    {
        int length = Math.Max(1, PositionsPerChunk(p));
        int half = p.Hidden / 2;
        double increment = Math.Log(10000.0) / (half - 1);
        var table = new float[length * p.Hidden];
        for (int pos = 0; pos < length; pos++)
        {
            for (int i = 0; i < half; i++)
            {
                float inverse = (float)Math.Exp(-increment * i);
                float angle = pos * inverse;
                table[(pos * p.Hidden) + i] = MathF.Sin(angle);
                table[(pos * p.Hidden) + half + i] = MathF.Cos(angle);
            }
        }

        return b.Constant("audio.position_table", DataType.F32, new Shape(length, p.Hidden), MemoryMarshal.AsBytes(table.AsSpan()).ToArray());
    }

    private static int PositionsPerChunk(AudioTowerParameters p) => new Speech.ChunkedAudioLayout(p.ChunkFrames, p.WindowFrames, p.ConvCount, p.MelBins).RowsPerChunk;
}
