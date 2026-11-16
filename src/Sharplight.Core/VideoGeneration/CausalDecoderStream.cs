using Sharplight.Imaging;
using Sharplight.Runtime;
using Sharplight.Vision;

namespace Sharplight.VideoGeneration;

/// <summary>
/// Streams a causal video VAE decoder over latent frames (<see cref="VideoDiffusionGraphs.VaeDecodeFirst"/>, then
/// <see cref="VideoDiffusionGraphs.VaeDecodeNext"/> per further latent frame), keeping per cache name the last two input frames of
/// each causal convolution (zeros before the first) between the runs. One instance per video.
/// </summary>
internal sealed class CausalDecoderStream
{
    private readonly Dictionary<string, float[]> _caches = new(StringComparer.Ordinal);

    /// <summary>Decodes one latent frame into planar frames [n, 3, H, W] in [0, 1].</summary>
    /// <param name="decoder">The first-frame decoder for the first call, the next-frame decoder after it.</param>
    /// <param name="latentFrame">The latent frame [C, 1, h, w] in model space.</param>
    /// <returns>The decoded frames' values and their count.</returns>
    public (float[] Pixels, int Frames) Decode(CompiledModel decoder, float[] latentFrame)
    {
        decoder.SetInput<float>(decoder.Graph.Inputs.Single(i => i.Name == VideoDiffusionGraphs.Latent), latentFrame);
        foreach (var input in decoder.Graph.Inputs.Where(i => i.Name.StartsWith(VideoDiffusionGraphs.CacheInPrefix, StringComparison.Ordinal)))
        {
            string name = input.Name[VideoDiffusionGraphs.CacheInPrefix.Length..];
            var cache = _caches.TryGetValue(name, out var stored) ? stored : new float[input.Shape.Elements()];
            decoder.SetInput<float>(input, cache);
        }

        var parameters = new RunParameters();
        decoder.Run(parameters);
        foreach (string output in decoder.Graph.Outputs.Keys)
        {
            if (!output.StartsWith(VideoDiffusionGraphs.CacheOutPrefix, StringComparison.Ordinal))
            {
                continue;
            }

            string name = output[VideoDiffusionGraphs.CacheOutPrefix.Length..];
            var shape = decoder.OutputShape(output, parameters);
            var fresh = new float[shape.Aggregate(1L, (a, b) => a * b)];
            decoder.GetOutput<float>(output, fresh);
            _caches[name] = KeepLastTwo(_caches.GetValueOrDefault(name), fresh, fresh.Length / (int)shape[0]);
        }

        var frameShape = decoder.OutputShape(VideoDiffusionGraphs.Frames, parameters);
        var pixels = new float[frameShape.Aggregate(1L, (a, b) => a * b)];
        decoder.GetOutput<float>(VideoDiffusionGraphs.Frames, pixels);
        return (pixels, (int)frameShape[0]);
    }

    /// <summary>Decodes one latent frame into 8-bit RGB frames.</summary>
    /// <param name="decoder">The first-frame decoder for the first call, the next-frame decoder after it.</param>
    /// <param name="latentFrame">The latent frame [C, 1, h, w] in model space.</param>
    /// <param name="width">Frame width.</param>
    /// <param name="height">Frame height.</param>
    public List<ImageInput> DecodeImages(CompiledModel decoder, float[] latentFrame, int width, int height)
    {
        var (pixels, count) = Decode(decoder, latentFrame);
        int frameValues = 3 * width * height;
        var frames = new List<ImageInput>(count);
        for (int f = 0; f < count; f++)
        {
            frames.Add(PlanarImage.FromPlanar(pixels.AsSpan(f * frameValues, frameValues), width, height, 0f, 1f));
        }

        return frames;
    }

    /// <summary>The last two frames of the previous cache (zeros when absent) followed by the fresh frames.</summary>
    private static float[] KeepLastTwo(float[]? previous, float[] fresh, int frameValues)
    {
        int freshFrames = fresh.Length / frameValues;
        if (freshFrames >= 2)
        {
            return fresh.AsSpan(fresh.Length - 2 * frameValues).ToArray();
        }

        var cache = new float[2 * frameValues];
        if (previous is not null)
        {
            previous.AsSpan(previous.Length - frameValues).CopyTo(cache);
        }

        fresh.CopyTo(cache.AsSpan(frameValues));
        return cache;
    }
}
