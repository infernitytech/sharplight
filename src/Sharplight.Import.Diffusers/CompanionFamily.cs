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

    /// <summary>Qwen-Image: a Qwen2.5-VL-7B language model (GGUF or Hugging Face folder), the causal 3-D (Wan 2.1-layout) VAE.</summary>
    QwenImage,

    /// <summary>Qwen-Image 2.1: a Qwen3-VL-8B language model (GGUF or Hugging Face folder), the residual 16× VAE.</summary>
    QwenImage21,

    /// <summary>FLUX.2: a Qwen3 language model (GGUF or Hugging Face folder), the 32-channel KL VAE with batch-norm statistics.</summary>
    Flux2,

    /// <summary>FLUX.2 Dev: Mistral Small 3.2 24B (GGUF, architecture <c>llama</c>, or the pipeline's <c>text_encoder</c> folder), the FLUX.2 VAE.</summary>
    Flux2Dev,

    /// <summary>Krea 2: a Qwen3-VL-4B language model (GGUF or Hugging Face folder), the Qwen-Image VAE.</summary>
    Krea2,
}
