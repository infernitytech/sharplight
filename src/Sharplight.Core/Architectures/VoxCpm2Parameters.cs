using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>The sizes of the VoxCPM2 parts the <see cref="VoxCpm2Adapter"/> builds itself (the language models are the decoder adapter's).</summary>
/// <param name="Hidden">The language models' width.</param>
/// <param name="PatchSize">Latent frames per patch (one language model step).</param>
/// <param name="FeatureSize">The width of an AudioVAE latent frame.</param>
/// <param name="QuantizerLatent">The scalar quantizer's latent width.</param>
/// <param name="QuantizerScale">The scalar quantizer's levels per side (values are rounded to multiples of 1 / scale).</param>
/// <param name="Encoder">The local feature encoder.</param>
/// <param name="Dit">The local diffusion transformer.</param>
/// <param name="NormEps">The RMS norm epsilon of the local transformers.</param>
/// <param name="RopeBase">The rotary base of the local transformers.</param>
/// <param name="Vae">The AudioVAE.</param>
internal sealed record VoxCpm2Parameters(int Hidden, int PatchSize, int FeatureSize, int QuantizerLatent, int QuantizerScale, VoxCpm2TransformerParameters Encoder,
    VoxCpm2TransformerParameters Dit, float NormEps, float RopeBase, AudioVaeV2Parameters Vae)
{
    /// <summary>Reads the parameters of the canonical source.</summary>
    /// <param name="src">The source (<c>voxcpm2.*</c> keys, the language model's <c>decoder.hidden_size</c>).</param>
    public static VoxCpm2Parameters From(ModelSource src)
    {
        const string A = VoxCpm2Adapter.Architecture + ".";
        return new VoxCpm2Parameters(
            Hidden: src.Get<int>("decoder.hidden_size"),
            PatchSize: src.Get<int>(A + "patch_size"),
            FeatureSize: src.Get<int>(A + "feat_dim"),
            QuantizerLatent: src.Get<int>(A + "fsq.latent"),
            QuantizerScale: src.Get<int>(A + "fsq.scale"),
            Encoder: VoxCpm2TransformerParameters.From(src, A + "encoder."),
            Dit: VoxCpm2TransformerParameters.From(src, A + "dit."),
            NormEps: src.Get<float>(A + "norm_eps"),
            RopeBase: src.Get<float>(A + "rope.base"),
            Vae: AudioVaeV2Parameters.From(src, A + "vae."));
    }
}
