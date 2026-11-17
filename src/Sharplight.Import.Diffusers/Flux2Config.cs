using System.Text.Json;
using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// The configuration of a FLUX.2 transformer and its pipeline defaults, written as the canonical <c>flux2.transformer.*</c> (the FLUX
/// transformer keys with shared modulations, SwiGLU and 4-axis RoPE), <c>flux2.scheduler.*</c> (FLUX.2's empirical dynamic shift),
/// text-encoder (Klein: Qwen3's hidden states entering layers 9, 18 and 27 over the chat-templated prompt of 512 tokens; Dev: Mistral Small
/// 3.2 24B's entering layers 10, 20 and 30 over the system-templated prompt) and default keys (distilled Klein: 4 steps without guidance;
/// Klein base models: 50 steps, guidance 4; the guidance-distilled Dev: 50 steps with the embedded guidance 4, no classifier-free guidance).
/// </summary>
internal sealed record Flux2Config
{
    /// <summary>Canonical architecture id.</summary>
    public const string Architecture = "flux2";

    /// <summary>Prefix of the architecture's metadata keys.</summary>
    public const string Prefix = Architecture + ".";

    /// <summary>The Qwen3 chat template of one user turn with thinking disabled (an empty think block), as the pipeline applies it.</summary>
    public const string PromptTemplate = "<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

    /// <summary>
    /// FLUX.2 Dev's Mistral Small 3.2 template (the pipeline's system message, one user turn, no generation prompt), as
    /// <c>apply_chat_template</c> renders it.
    /// </summary>
    public const string DevPromptTemplate = "<s>[SYSTEM_PROMPT]You are an AI that reasons about image descriptions. You give structured responses focusing on object "
        + "relationships, object\nattribution and actions without speculation.[/SYSTEM_PROMPT][INST]{prompt}[/INST]";

    /// <summary>Prompt tokens of the pipeline.</summary>
    public const int PromptLength = 512;

    /// <summary>Model width.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Attention heads.</summary>
    public required int HeadCount { get; init; }

    /// <summary>Dual-stream blocks.</summary>
    public required int DoubleLayers { get; init; }

    /// <summary>Single-stream blocks.</summary>
    public required int SingleLayers { get; init; }

    /// <summary>Text feature width (three Qwen3 hidden states).</summary>
    public required int ContextDim { get; init; }

    /// <summary>Latent channels of the transformer (the patchified VAE latents).</summary>
    public required int LatentChannels { get; init; }

    /// <summary>Whether the guidance scale is embedded.</summary>
    public required bool GuidanceEmbed { get; init; }

    /// <summary>Whether the model is guidance-distilled with a Mistral text encoder (FLUX.2 Dev: the guidance embedding).</summary>
    public bool Dev => GuidanceEmbed;

    /// <summary>Whether the model is step-distilled (Klein: 4 steps, no guidance).</summary>
    public bool Distilled { get; init; } = true;

    /// <summary>RoPE dimensions per axis.</summary>
    public int[] AxesDim { get; init; } = [32, 32, 32, 32];

    /// <summary>RoPE base.</summary>
    public float RopeTheta { get; init; } = 2000f;

    /// <summary>The original scheduler configuration, or null.</summary>
    public string? SchedulerJson { get; init; }

    /// <summary>Infers the configuration from canonical <c>transformer.*</c> weights (128-wide heads).</summary>
    /// <param name="weights">The canonical weights.</param>
    /// <exception cref="InvalidDataException">The weights are not a FLUX.2 transformer.</exception>
    public static Flux2Config FromWeights(IReadOnlyDictionary<string, WeightInfo> weights)
    {
        if (!weights.TryGetValue("transformer.img_in.weight", out var embedder) || !weights.TryGetValue("transformer.double.0.img_attn.norm_q.weight", out var norm))
        {
            throw new InvalidDataException("The checkpoint has no FLUX.2 transformer.");
        }

        int hidden = (int)embedder.Shape[0].Fixed;
        int headDim = (int)norm.Shape[0].Fixed;
        return new Flux2Config
        {
            // FLUX.2 rotates four equal quarters of each head.
            AxesDim = [headDim / 4, headDim / 4, headDim / 4, headDim / 4],
            HiddenSize = hidden,
            HeadCount = hidden / headDim,
            DoubleLayers = FluxNames.CountBlocks(weights, "double"),
            SingleLayers = FluxNames.CountBlocks(weights, "single"),
            ContextDim = (int)weights["transformer.txt_in.weight"].Shape[1].Fixed,
            LatentChannels = (int)embedder.Shape[1].Fixed,
            GuidanceEmbed = weights.ContainsKey("transformer.guidance_in.in_layer.weight"),
        };
    }

    /// <summary>This configuration with the settings of a diffusers folder.</summary>
    /// <param name="schedulerJson">The <c>scheduler_config.json</c> text, or null.</param>
    /// <param name="transformerConfig">The <c>transformer/config.json</c>, or null.</param>
    /// <param name="modelIndex">The <c>model_index.json</c>, or null.</param>
    public Flux2Config WithDiffusers(string? schedulerJson, JsonElement? transformerConfig, JsonElement? modelIndex)
    {
        var result = this with { SchedulerJson = schedulerJson };
        if (transformerConfig is { } config)
        {
            result = result with
            {
                AxesDim = config.TryGetProperty("axes_dims_rope", out _) ? config.Ints("axes_dims_rope") : AxesDim,
                RopeTheta = config.Float("rope_theta", RopeTheta),
            };
        }

        return modelIndex is { } index ? result with { Distilled = index.Bool("is_distilled", false) } : result;
    }

    /// <summary>Writes the transformer, scheduler, family and default keys.</summary>
    /// <param name="metadata">The canonical metadata.</param>
    public void WriteTo(IDictionary<string, object> metadata)
    {
        string t = Prefix + "transformer.";
        metadata[t + "latent_channels"] = LatentChannels;
        metadata[t + "patch_size"] = 1;
        metadata[t + "hidden_size"] = HiddenSize;
        metadata[t + "head_count"] = HeadCount;
        metadata[t + "double_layers"] = DoubleLayers;
        metadata[t + "single_layers"] = SingleLayers;
        metadata[t + "context_dim"] = ContextDim;
        metadata[t + "pooled_dim"] = 0;
        metadata[t + "guidance_embed"] = GuidanceEmbed;
        metadata[t + "axes_dim"] = AxesDim;
        metadata[t + "rope_theta"] = RopeTheta;
        metadata[t + "shared_modulation"] = true;
        metadata[t + "swiglu"] = true;
        metadata[t + "text_position_axis"] = 3;
        // Reference images (editing): the pipelines' time coordinate 10 + 10 · i (Flux2Pipeline._prepare_image_ids), up to ten images.
        metadata[t + "max_references"] = 10;
        metadata[t + "reference_index_base"] = 10;
        metadata[t + "reference_index_step"] = 10;
        metadata[t + "sample_size"] = 1024;
        string s = Prefix + "scheduler.";
        metadata[s + "train_steps"] = 1000;
        metadata[s + "shift"] = 3f;
        metadata[s + "dynamic_shift"] = true;
        metadata[s + "empirical_shift"] = true;
        metadata[s + "sampler"] = "euler";
        if (SchedulerJson is not null)
        {
            metadata[s + "source_config"] = SchedulerJson;
        }

        metadata[Prefix + "text_encoder.prompt_length"] = PromptLength;
        metadata[Prefix + "text_encoder.layers"] = Dev ? new[] { 10, 20, 30 } : new[] { 9, 18, 27 };
        metadata[Prefix + "text_encoder.prompt_template"] = Dev ? DevPromptTemplate : PromptTemplate;
        // Mistral Small 3.2's hidden states carry outlier channels that int8-quantized activations lose (the padding rows, which the
        // transformer reads, most): its matrix products keep f32 activations.
        metadata[Prefix + "text_encoder.f32_activations"] = Dev;
        metadata[Prefix + "vae.latent_patch"] = 2;
        metadata[Prefix + "vae.batch_norm_eps"] = 1e-4f;
        metadata[Prefix + "vae.scaling_factor"] = 1f;
        metadata[Prefix + "vae.shift_factor"] = 0f;
        bool stepDistilled = Distilled && !Dev;
        metadata[Prefix + "default_steps"] = stepDistilled ? 4 : 50;
        metadata[Prefix + "default_guidance"] = stepDistilled ? 1.0 : 4.0;
        // Dev embeds the guidance scale instead of running classifier-free guidance.
        metadata[Prefix + "classifier_free_guidance"] = !Distilled && !Dev;
    }
}
