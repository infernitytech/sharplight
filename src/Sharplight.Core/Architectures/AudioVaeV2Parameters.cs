using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>The sizes of VoxCPM2's AudioVAE V2 (causal convolutional encoder at the input rate, causal convolutional decoder with sample-rate conditioning at the output rate).</summary>
/// <param name="EncoderDim">The encoder's first width (doubled by every downsampling block).</param>
/// <param name="EncoderRates">The encoder's strides.</param>
/// <param name="Latent">The latent width.</param>
/// <param name="DecoderDim">The decoder's first width (halved by every upsampling block).</param>
/// <param name="DecoderRates">The decoder's strides.</param>
/// <param name="SampleRate">The encoder's input rate.</param>
/// <param name="OutSampleRate">The decoder's output rate.</param>
/// <param name="SampleRateBucket">The row of the decoder's sample-rate conditioning tables for the output rate, or -1 without conditioning.</param>
internal sealed record AudioVaeV2Parameters(int EncoderDim, long[] EncoderRates, int Latent, int DecoderDim, long[] DecoderRates, int SampleRate, int OutSampleRate, int SampleRateBucket)
{
    /// <summary>The input samples per latent frame.</summary>
    public long EncoderHop => EncoderRates.Aggregate(1L, (a, r) => a * r);

    /// <summary>The output samples per latent frame.</summary>
    public long DecoderHop => DecoderRates.Aggregate(1L, (a, r) => a * r);

    /// <summary>Reads <c>{prefix}{encoder_dim,encoder_rates,latent,decoder_dim,decoder_rates,sample_rate,out_sample_rate,sample_rate_bucket}</c>.</summary>
    /// <param name="src">The source.</param>
    /// <param name="prefix">The key prefix, for example <c>voxcpm2.vae.</c>.</param>
    public static AudioVaeV2Parameters From(ModelSource src, string prefix)
        => new(src.Get<int>(prefix + "encoder_dim"), src.Get<long[]>(prefix + "encoder_rates"), src.Get<int>(prefix + "latent"), src.Get<int>(prefix + "decoder_dim"),
            src.Get<long[]>(prefix + "decoder_rates"), src.Get<int>(prefix + "sample_rate"), src.Get<int>(prefix + "out_sample_rate"), src.Get(prefix + "sample_rate_bucket", -1));
}
