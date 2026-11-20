using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// The hyper-parameters of the decoder of a SAME autoencoder (Stable Audio 3's transformer autoencoder) kept under a key prefix of another model: latents through a linear
/// layer and one resampling stage (each latent frame followed by <see cref="Stride"/> new tokens, transformer blocks over chunks or a sliding window, the new tokens
/// kept and mapped to patches of the waveform).
/// </summary>
/// <param name="Width">Width of the transformer.</param>
/// <param name="OutChannels">Channels of a waveform patch row (audio channels times the patch size).</param>
/// <param name="Stride">New tokens per latent frame.</param>
/// <param name="Layers">Transformer blocks.</param>
/// <param name="HeadDim">Width of a head.</param>
/// <param name="Differential">Whether the attention is differential.</param>
/// <param name="DynamicTanh">Whether the norms are DynamicTanh (else RMS norms).</param>
/// <param name="ChunkSize">Latent-side chunk size of the chunked attention (the chunks hold <c>ChunkSize / Stride</c> latent frames).</param>
/// <param name="MidpointShift">Whether the second half of the blocks works on chunks shifted by half a chunk.</param>
/// <param name="SlidingWindow">The latent frames to the left and right a token sees (sliding-window attention instead of chunks), or empty.</param>
/// <param name="SinusoidalBlocks">The last blocks whose feed-forward gate is <c>sin(π x)</c> instead of SiLU.</param>
/// <param name="MappingKernel">Kernel of the output convolution.</param>
/// <param name="PatchSize">Audio samples per patch row.</param>
/// <param name="MaskNoise">Standard deviation of the noise added to the new tokens.</param>
/// <param name="LatentNoise">Standard deviation of the noise added to the latents (relative to their scale).</param>
/// <param name="SoftClip">Whether the audio goes through tanh.</param>
/// <param name="DecodeFrames">Latent frames of one decoder run.</param>
/// <param name="DecodeOverlap">Frames of overlap between runs.</param>
internal sealed record SameDecoderParameters(int Width, int OutChannels, int Stride, int Layers, int HeadDim, bool Differential, bool DynamicTanh, int ChunkSize,
    bool MidpointShift, long[] SlidingWindow, int SinusoidalBlocks, int MappingKernel, int PatchSize, float MaskNoise, float LatentNoise, bool SoftClip, int DecodeFrames,
    int DecodeOverlap)
{
    /// <summary>Epsilon of the norms.</summary>
    public const float NormEps = 1e-3f;

    /// <summary>Rotary base.</summary>
    public const float RopeBase = 10000f;

    /// <summary>Rows of a sub-chunk: one latent frame and its new tokens.</summary>
    public int SubChunk => Stride + 1;

    /// <summary>Reads the parameters stored under <c>{architecture}.{prefix}</c>.</summary>
    /// <param name="source">The model's source.</param>
    /// <param name="prefix">The key prefix after the architecture (e.g. <c>vae.</c>).</param>
    public static SameDecoderParameters From(ModelSource source, string prefix) => new(
        source.Arch<int>(prefix + "width"), source.Arch<int>(prefix + "out_channels"), source.Arch<int>(prefix + "stride"), source.Arch<int>(prefix + "layer_count"),
        source.Arch<int>(prefix + "head_dim"), source.Arch(prefix + "differential", true), source.Arch(prefix + "dyt", true), source.Arch(prefix + "chunk_size", 128),
        source.Arch(prefix + "midpoint_shift", false), source.Arch(prefix + "sliding_window", Array.Empty<long>()), source.Arch(prefix + "sinusoidal_blocks", 0),
        source.Arch(prefix + "mapping_kernel", 1), source.Arch<int>(prefix + "patch_size"), source.Arch(prefix + "mask_noise", 0f), source.Arch(prefix + "latent_noise", 0f),
        source.Arch(prefix + "soft_clip", false), source.Arch(prefix + "decode_frames", 128), source.Arch(prefix + "decode_overlap", 32));
}
