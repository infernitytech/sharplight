namespace Sharplight.Models;

/// <summary>
/// Well-known component roles (<see cref="ModelComponent.Role"/>). Open-ended like the kinds: a model may use any lowercase,
/// hyphenated role; numbered variants (<c>text-encoder-2</c>) name further components of one kind.
/// </summary>
public static class ModelComponentRoles
{
    /// <summary>Autoregressive decoder: the whole model of a decoder-only LLM, the text decoder of a speech recognizer, the codec
    /// language model of a speech synthesizer.</summary>
    public const string Decoder = "decoder";

    /// <summary>Encoder of an encoder-decoder model (the audio encoder of a speech recognizer, the encoder of T5).</summary>
    public const string Encoder = "encoder";

    /// <summary>Tokenizer (vocabulary, merges, special tokens): usually a metadata-only component.</summary>
    public const string Tokenizer = "tokenizer";

    /// <summary>Turns raw input into model features (log-mel spectrogram parameters and filter banks, image normalization).</summary>
    public const string FeatureExtractor = "feature-extractor";

    /// <summary>Encodes a text prompt into conditioning (CLIP / T5 text encoders of image generators, the text frontend of TTS).</summary>
    public const string TextEncoder = "text-encoder";

    /// <summary>The denoising network of a diffusion model (UNet, DiT, MMDiT).</summary>
    public const string Denoiser = "denoiser";

    /// <summary>Variational autoencoder: maps latents to images (decoder graph) and images to latents (encoder graph).</summary>
    public const string Vae = "vae";

    /// <summary>The noise schedule and sampler configuration of a diffusion model: a configuration-only component.</summary>
    public const string Scheduler = "scheduler";

    /// <summary>Turns acoustic features or codec tokens into a waveform.</summary>
    public const string Vocoder = "vocoder";

    /// <summary>Image encoder (ViT trunk, CLIP / SigLIP vision tower, SAM image encoder); the text tower of a dual-tower model is the <see cref="TextEncoder"/>.</summary>
    public const string VisionEncoder = "vision-encoder";

    /// <summary>Encodes prompts (points, boxes, masks) of a promptable segmenter.</summary>
    public const string PromptEncoder = "prompt-encoder";

    /// <summary>Turns encoder features (and prompt embeddings) into masks, boxes or per-pixel maps.</summary>
    public const string MaskDecoder = "mask-decoder";
}
