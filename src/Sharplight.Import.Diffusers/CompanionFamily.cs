namespace Sharplight.Import.Diffusers;

/// <summary>The model families whose transformer-only files need companion files (<see cref="DiffusionCompanions"/>).</summary>
internal enum CompanionFamily
{
    /// <summary>FLUX.1: CLIP-L, T5-XXL, the 16-channel VAE.</summary>
    Flux,

    /// <summary>SD3 / SD3.5: CLIP-L, CLIP-G, T5-XXL, the 16-channel VAE.</summary>
    Sd3,

    /// <summary>Chroma: T5-XXL (its only text encoder), the 16-channel VAE.</summary>
    Chroma,

    /// <summary>Z-Image: a Qwen3-4B language model (GGUF or Hugging Face folder), the 16-channel VAE.</summary>
    ZImage,
}
