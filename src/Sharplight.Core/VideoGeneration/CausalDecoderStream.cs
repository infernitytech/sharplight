using Sharplight.Imaging;
using Sharplight.Runtime;
using Sharplight.Vision;

namespace Sharplight.VideoGeneration;

/// <summary>
/// Streams a causal video VAE over chunks: the decoder over latent frames (<see cref="VideoDiffusionGraphs.VaeDecodeFirst"/>, then
/// <see cref="VideoDiffusionGraphs.VaeDecodeNext"/> per further latent frame) or the encoder over frames
/// (<see cref="VideoDiffusionGraphs.VaeEncode"/>, then <see cref="VideoDiffusionGraphs.VaeEncodeNext"/>), keeping per cache name the last
/// two input frames of each causal convolution (zeros before the first) between the runs; a cache input of one frame gets the last one.
/// One instance per video and direction.
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
        var parameters = RunChunk(decoder, VideoDiffusionGraphs.Latent, latentFrame);
        var frameShape = decoder.OutputShape(VideoDiffusionGraphs.Frames, parameters);
        var pixels = new float[frameShape.Aggregate(1L, (a, b) => a * b)];
        decoder.GetOutput<float>(VideoDiffusionGraphs.Frames, pixels);
        return (pixels, (int)frameShape[0]);
    }

    /// <summary>Encodes one chunk of frames (the first image, or temporal-compression frames) into the next latent frame.</summary>
    /// <param name="encoder">The first-chunk encoder for the first call, the next-chunk encoder after it.</param>
    /// <param name="inputName">The input: <see cref="VideoDiffusionGraphs.Image"/> [3, H, W] or <see cref="VideoDiffusionGraphs.Frames"/> [3, r, H, W].</param>
    /// <param name="pixels">The chunk's pixels in [0, 1].</param>
    /// <returns>The normalized latent frame [C, h, w].</returns>
    public float[] Encode(CompiledModel encoder, string inputName, float[] pixels)
    {
        var parameters = RunChunk(encoder, inputName, pixels);
        var latent = new float[encoder.OutputShape(VideoDiffusionGraphs.Latent, parameters).Aggregate(1L, (a, b) => a * b)];
        encoder.GetOutput<float>(VideoDiffusionGraphs.Latent, latent);
        return latent;
    }

    /// <summary>Runs one chunk with the carried caches and keeps the new ones.</summary>
    private RunParameters RunChunk(CompiledModel graph, string inputName, float[] values)
    {
        graph.SetInput<float>(graph.Graph.Inputs.Single(i => i.Name == inputName), values);
        foreach (var input in graph.Graph.Inputs.Where(i => i.Name.StartsWith(VideoDiffusionGraphs.CacheInPrefix, StringComparison.Ordinal)))
        {
            string name = input.Name[VideoDiffusionGraphs.CacheInPrefix.Length..];
            long wanted = input.Shape.Elements();
            var cache = _caches.TryGetValue(name, out var stored) ? stored.AsSpan(stored.Length - (int)wanted).ToArray() : new float[wanted];
            graph.SetInput<float>(input, cache);
        }

        var parameters = new RunParameters();
        graph.Run(parameters);
        foreach (string output in graph.Graph.Outputs.Keys)
        {
            if (!output.StartsWith(VideoDiffusionGraphs.CacheOutPrefix, StringComparison.Ordinal))
            {
                continue;
            }

            string name = output[VideoDiffusionGraphs.CacheOutPrefix.Length..];
            var shape = graph.OutputShape(output, parameters);
            var fresh = new float[shape.Aggregate(1L, (a, b) => a * b)];
            graph.GetOutput<float>(output, fresh);
            _caches[name] = KeepLastTwo(_caches.GetValueOrDefault(name), fresh, fresh.Length / (int)shape[0]);
        }

        return parameters;
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
