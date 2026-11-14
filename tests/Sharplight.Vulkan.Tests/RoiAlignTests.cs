using Sharplight.Graph;

namespace Sharplight.Vulkan.Tests;

/// <summary>RoiAlign on the GPU against the CPU oracle: avg and max, both coordinate transformations, fixed and adaptive sampling.</summary>
public sealed class RoiAlignTests
{
    /// <summary>RoIs in input coordinates, partly outside the feature map, some degenerate (zero or negative size).</summary>
    private static float[] Rois(long count, int seed, float width, float height)
    {
        var rng = new Random(seed);
        var rois = new float[count * 4];
        for (long i = 0; i < count; i++)
        {
            float x1 = (float)(rng.NextDouble() * 1.2 - 0.1) * width;
            float y1 = (float)(rng.NextDouble() * 1.2 - 0.1) * height;
            float w = i % 11 == 0 ? 0f : (float)rng.NextDouble() * width * 0.6f;
            float h = i % 13 == 0 ? -1f : (float)rng.NextDouble() * height * 0.6f;
            (rois[i * 4], rois[i * 4 + 1], rois[i * 4 + 2], rois[i * 4 + 3]) = (x1, y1, x1 + w, y1 + h);
        }

        return rois;
    }

    [Theory]
    [InlineData("avg", true, 0, 0.25f, 7, 7)]
    [InlineData("avg", false, 0, 0.25f, 7, 7)]
    [InlineData("max", true, 2, 0.25f, 5, 3)]
    [InlineData("max", false, 0, 0.5f, 4, 4)]
    [InlineData("avg", true, 2, 1f, 14, 14)]
    public void Modes(string mode, bool halfPixel, long samplingRatio, float spatialScale, long outH, long outW)
    {
        const int Batch = 2;
        const int Channels = 5;
        const int Height = 20;
        const int Width = 24;
        var tg = new TG();
        var x = tg.InF("x", Batch, Channels + 2, Height, Width);
        var features = tg.B.Slice(x, 1, 1, Channels);   // strided feature map
        var rois = tg.InF("rois", TG.T, 4);
        var batch = tg.In("batch", Tensors.DataType.I32, TG.T);
        var attributes = Attributes.From(("output_height", outH), ("output_width", outW), ("sampling_ratio", samplingRatio), ("spatial_scale", spatialScale),
            ("mode", mode), ("half_pixel", halfPixel));
        tg.B.Output("y", tg.B.Emit1("RoiAlign", [features, rois, batch], attributes));
        var graph = tg.B.Build();
        float extent = 1f / spatialScale;
        var steps = new List<Step>();
        foreach (int count in new[] { 37, 5 })
        {
            steps.Add(new Step(count, 0, (x, Rnd.F((long)Batch * (Channels + 2) * Height * Width, 3)), (rois, Rois(count, count, Width * extent, Height * extent)),
                (batch, Rnd.I(count, 9, Batch))));
        }

        Diff.Check("RoiAlign", graph, steps, atol: 1e-5, rtol: 1e-4, what: $"{mode} half_pixel={halfPixel} ratio={samplingRatio} scale={spatialScale} {outH}x{outW}");
    }
}
