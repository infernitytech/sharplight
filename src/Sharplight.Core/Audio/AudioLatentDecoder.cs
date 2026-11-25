using Sharplight.Models;
using Sharplight.Runtime;

namespace Sharplight.Audio;

/// <summary>
/// Runs the decoder graph of an audio autoencoder (<see cref="AudioGenerationGraphs.DecodeGraph"/>) over latents in overlapping windows: each window of
/// <c>windowFrames</c> frames is decoded with <c>overlap</c> frames of context on either side and only its own samples are kept, so long audio decodes in bounded memory
/// with the result of a decoder whose receptive field the overlap covers.
/// </summary>
internal static class AudioLatentDecoder
{
    /// <summary>Decodes the first <paramref name="frames"/> frames of <paramref name="latents"/> (the frames after them serve as context).</summary>
    /// <param name="decode">The compiled decoder graph.</param>
    /// <param name="latents">Frame-major latents, <paramref name="latentChannels"/> values per frame.</param>
    /// <param name="totalFrames">The frames of <paramref name="latents"/>.</param>
    /// <param name="frames">The frames to decode.</param>
    /// <param name="latentChannels">The values per latent frame.</param>
    /// <param name="samplesPerFrame">The audio samples per latent frame.</param>
    /// <param name="audioChannels">The audio channels.</param>
    /// <param name="windowFrames">The frames one decoder run covers.</param>
    /// <param name="overlap">The frames of context on either side of a window.</param>
    /// <param name="cancellationToken">Stops between windows.</param>
    /// <returns>Interleaved samples, <c>frames · samplesPerFrame · audioChannels</c> values.</returns>
    public static float[] Decode(CompiledModel decode, ReadOnlySpan<float> latents, int totalFrames, int frames, int latentChannels, int samplesPerFrame, int audioChannels, int windowFrames,
        int overlap, CancellationToken cancellationToken)
    {
        var samples = new float[frames * samplesPerFrame * audioChannels];
        var input = decode.Graph.Inputs.First(v => v.Input!.Name == AudioGenerationGraphs.LatentInput);
        for (int start = 0; start < frames; start += windowFrames)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int end = Math.Min(frames, start + windowFrames);
            int from = Math.Max(0, start - overlap);
            int to = Math.Min(totalFrames, end + overlap);
            var run = new RunParameters();
            run.Symbols[WellKnown.TokenSymbol] = to - from;
            decode.SetInput<float>(input, latents.Slice(from * latentChannels, (to - from) * latentChannels));
            decode.Run(run);
            var window = new float[(to - from) * samplesPerFrame * audioChannels];
            decode.GetOutput<float>(AudioGenerationGraphs.AudioOutput, window);
            window.AsSpan((start - from) * samplesPerFrame * audioChannels, (end - start) * samplesPerFrame * audioChannels)
                .CopyTo(samples.AsSpan(start * samplesPerFrame * audioChannels));
        }

        return samples;
    }

    /// <summary>Scales the samples down so that the peak is at most 1 (nothing changes when it already is).</summary>
    /// <param name="samples">The samples, changed in place.</param>
    public static void LimitPeak(Span<float> samples)
    {
        float peak = 0f;
        foreach (float v in samples)
        {
            peak = Math.Max(peak, Math.Abs(v));
        }

        if (peak > 1f)
        {
            for (int i = 0; i < samples.Length; i++)
            {
                samples[i] /= peak;
            }
        }
    }
}
