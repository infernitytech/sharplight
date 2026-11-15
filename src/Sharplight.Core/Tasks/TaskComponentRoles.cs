namespace Sharplight.Tasks;

/// <summary>
/// Well-known component roles of multi-component models (<see cref="TaskComponent.Role"/>). Open-ended like the task ids: a model
/// may use any lowercase, underscore-separated role; numbered variants (<c>text_encoder_2</c>) name further components of one kind.
/// </summary>
public static class TaskComponentRoles
{
    /// <summary>Encodes a text prompt into conditioning (CLIP / T5 text encoders of image generators, the text frontend of TTS).</summary>
    public const string TextEncoder = "text_encoder";

    /// <summary>Encodes audio features (the encoder of a speech recognizer).</summary>
    public const string AudioEncoder = "audio_encoder";

    /// <summary>Autoregressive token decoder (the text decoder of a speech recognizer, the codec language model of a TTS model).</summary>
    public const string Decoder = "decoder";

    /// <summary>The denoising network of a diffusion model (UNet, DiT, MMDiT).</summary>
    public const string Denoiser = "denoiser";

    /// <summary>Maps images to latents (image-to-image, inpainting).</summary>
    public const string VaeEncoder = "vae_encoder";

    /// <summary>Maps latents to images.</summary>
    public const string VaeDecoder = "vae_decoder";

    /// <summary>Turns acoustic features or codec tokens into a waveform.</summary>
    public const string Vocoder = "vocoder";
}
