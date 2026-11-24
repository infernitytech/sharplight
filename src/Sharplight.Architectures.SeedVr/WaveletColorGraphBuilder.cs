using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;
using Sharplight.VideoGeneration;

namespace Sharplight.Architectures.SeedVr;

/// <summary>
/// Builds the wavelet colour transfer (<see cref="SeedVrGraphs.ColorCorrect"/>) of SeedVR2's inference scripts for one clip size: the
/// result keeps the high frequencies of the restored frames and takes the low frequencies of the input, where the low frequencies are five
/// successive 3×3 binomial blurs with dilations 1, 2, 4, 8, 16 and edge-replicating borders. The blur is linear, so content − L(content) +
/// L(input) is computed as content + L(input − content). The result is mapped from [−1, 1] to [0, 1] and clamped.
/// </summary>
internal static class WaveletColorGraphBuilder
{
    private const int Levels = 5;

    /// <summary>Builds the graph.</summary>
    /// <param name="frames">Frames of the clip.</param>
    /// <param name="height">Height in pixels.</param>
    /// <param name="width">Width in pixels.</param>
    public static ComputeGraph Build(int frames, int height, int width)
    {
        var b = new GraphBuilder(SeedVrGraphs.ColorCorrect, _ => null);
        var shape = new Shape(frames, 3, height, width);
        var content = b.Input(VideoDiffusionGraphs.Frames, InputRole.Other, DataType.F32, shape);
        var reference = b.Input(SeedVrGraphs.Reference, InputRole.Other, DataType.F32, shape);
        float[] taps = [0.0625f, 0.125f, 0.0625f, 0.125f, 0.25f, 0.125f, 0.0625f, 0.125f, 0.0625f];
        var kernel = new float[3 * taps.Length];
        for (int c = 0; c < 3; c++)
        {
            taps.CopyTo(kernel, c * taps.Length);
        }

        var weight = b.Constant("binomial_kernel", DataType.F32, new Shape(3, 1, 3, 3), MemoryMarshal.AsBytes(kernel.AsSpan()).ToArray());
        var low = b.Sub(reference, content);
        for (int level = 0; level < Levels; level++)
        {
            long radius = 1L << level;
            var padded = b.Emit1("Pad", [low], Attributes.From(("pads", new long[] { 0, 0, radius, radius, 0, 0, radius, radius }), ("mode", "edge")));
            low = b.Emit1("Conv", [padded, weight], Attributes.From(("dilations", new[] { radius, radius }), ("groups", 3L)));
        }

        var corrected = b.Scale(b.Add(content, low), 0.5f, 0.5f);
        b.Output(SeedVrGraphs.Output, b.Emit1("Clip", [corrected], Attributes.From(("min", 0f), ("max", 1f))));
        return b.Build();
    }
}
