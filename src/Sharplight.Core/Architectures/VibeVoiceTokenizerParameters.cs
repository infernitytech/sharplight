using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// The hyper-parameters of one VibeVoice speech tokenizer (the acoustic one: 64-wide continuous latents at 7.5 Hz, encoder and decoder; the semantic one: 128-wide, encoder only): a
/// stack of causal ConvNeXt-style stages (depthwise convolution, RMS norm, feed-forward, layer scale) joined by strided convolutions (encoder) or transposed convolutions (decoder).
/// </summary>
/// <param name="Filters">Channels of the stem (doubled after every downsampling).</param>
/// <param name="Kernel">Kernel of the stem, the head and the depthwise convolutions.</param>
/// <param name="Depths">Blocks per encoder stage, from the audio rate to the frame rate; the decoder's stages run in reverse.</param>
/// <param name="Ratios">The upsampling ratio of each decoder step, in the decoder's order (the encoder downsamples with them in reverse).</param>
/// <param name="Latent">The width of the latents.</param>
/// <param name="Eps">The RMS norm epsilon.</param>
internal sealed record VibeVoiceTokenizerParameters(int Filters, int Kernel, int[] Depths, long[] Ratios, int Latent, float Eps)
{
    /// <summary>The audio samples one latent frame covers.</summary>
    public long Hop => Ratios.Aggregate(1L, (a, b) => a * b);

    /// <summary>The number of stages (one more than the ratios: the stem's stage runs at the audio rate).</summary>
    public int Stages => Depths.Length;

    /// <summary>Reads the parameters under <paramref name="prefix"/> (for example <c>vibevoice.acoustic.</c>) of a canonical source.</summary>
    /// <param name="source">The source.</param>
    /// <param name="prefix">The key prefix, with its trailing dot.</param>
    public static VibeVoiceTokenizerParameters From(ModelSource source, string prefix) => new(
        source.Get<int>(prefix + "filters"), source.Get<int>(prefix + "kernel"), source.Get<long[]>(prefix + "depths").Select(d => (int)d).ToArray(),
        source.Get<long[]>(prefix + "ratios"), source.Get<int>(prefix + "latent"), source.Get<float>(prefix + "eps"));
}
