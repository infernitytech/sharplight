namespace Sharplight.Vision;

/// <summary>Auxiliary source roles understood by adapters.</summary>
public static class AuxiliaryRoles
{
    /// <summary>A separate vision tower file (llama.cpp's "mmproj").</summary>
    public const string Mmproj = "mmproj";

    /// <summary>A speculative-decoding drafter trained for the primary model (an EAGLE-3 checkpoint); it becomes the model's draft graph.</summary>
    public const string Draft = "draft";

    /// <summary>The decoder of an audio codec that turns the codec tokens a speech language model generates into a waveform (OuteTTS's WavTokenizer).</summary>
    public const string Vocoder = "vocoder";

    /// <summary>
    /// The (first) text encoder of an image generator distributed without it (FLUX's CLIP-L next to a transformer-only file); the same
    /// name as the component role.
    /// </summary>
    public const string TextEncoder = "text-encoder";

    /// <summary>The second text encoder of an image generator distributed without it (FLUX's T5-XXL, SD3's CLIP-G).</summary>
    public const string SecondTextEncoder = "text-encoder-2";

    /// <summary>The third text encoder of an image generator distributed without it (SD3's T5-XXL).</summary>
    public const string ThirdTextEncoder = "text-encoder-3";

    /// <summary>The autoencoder of an image generator distributed without it (FLUX's 16-channel VAE).</summary>
    public const string Vae = "vae";

    /// <summary>A second denoiser of a two-expert model (Wan 2.2 A14B: the low-noise transformer beside the high-noise one).</summary>
    public const string SecondTransformer = "transformer-2";

    /// <summary>The image encoder whose states condition a video model (Wan 2.1 I2V / FLF2V: CLIP ViT-H/14 with its image processor).</summary>
    public const string ImageEncoder = "image-encoder";
}
