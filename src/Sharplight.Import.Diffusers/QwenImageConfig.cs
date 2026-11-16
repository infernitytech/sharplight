using System.Text.Json;
using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// The configuration of a Qwen-Image transformer and its pipeline defaults, written as the canonical <c>qwen_image.transformer.*</c>
/// (in the FLUX transformer's keys: dual-stream blocks only, centred RoPE, masked text), <c>qwen_image.scheduler.*</c>, text-encoder and
/// default keys (Qwen-Image: the exponential dynamic shift between 0.5 and 0.9 with the terminal stretch to 0.02, 50 steps, guidance 4
/// with the norm rescaling; the Qwen2.5-VL encoder's final-normed last hidden states over the system-templated prompt, its 34 template
/// tokens dropped, at most 512 tokens).
/// </summary>
internal sealed record QwenImageConfig
{
    /// <summary>Canonical architecture id.</summary>
    public const string Architecture = "qwen_image";

    /// <summary>Prefix of the architecture's metadata keys.</summary>
    public const string Prefix = Architecture + ".";

    /// <summary>The pipeline's prompt template (a system turn describing the image, then the prompt as the user turn).</summary>
    public const string PromptTemplate = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships "
        + "of the objects and background:<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n";

    /// <summary>Tokens of the template before the prompt whose hidden states the pipeline drops.</summary>
    public const int PromptSkip = 34;

    /// <summary>Prompt tokens the transformer sees (the pipeline's <c>max_sequence_length</c> default).</summary>
    public const int PromptLength = 512;

    /// <summary>Model width.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Attention heads.</summary>
    public required int HeadCount { get; init; }

    /// <summary>Dual-stream blocks.</summary>
    public required int Layers { get; init; }

    /// <summary>Text feature width.</summary>
    public required int ContextDim { get; init; }

    /// <summary>Patch input width (latent channels × patch area).</summary>
    public required int InChannels { get; init; }

    /// <summary>RoPE dimensions per position axis.</summary>
    public int[] AxesDim { get; init; } = [16, 56, 56];

    /// <summary>Latent channels.</summary>
    public int LatentChannels { get; init; } = 16;

    /// <summary>The original scheduler configuration, or null.</summary>
    public string? SchedulerJson { get; init; }

    /// <summary>Infers the configuration from canonical <c>transformer.*</c> weights (128-wide heads, as every release).</summary>
    /// <param name="weights">The canonical weights.</param>
    /// <exception cref="InvalidDataException">The weights are not a Qwen-Image transformer.</exception>
    public static QwenImageConfig FromWeights(IReadOnlyDictionary<string, WeightInfo> weights)
    {
        if (!weights.TryGetValue("transformer.img_in.weight", out var embedder) || !weights.TryGetValue("transformer.double.0.img_attn.norm_q.weight", out var norm))
        {
            throw new InvalidDataException("The checkpoint has no Qwen-Image transformer.");
        }

        int hidden = (int)embedder.Shape[0].Fixed;
        int headDim = (int)norm.Shape[0].Fixed;
        int layers = 0;
        while (weights.ContainsKey($"transformer.double.{layers}.img_attn.norm_q.weight"))
        {
            layers++;
        }

        return new QwenImageConfig
        {
            HiddenSize = hidden,
            HeadCount = hidden / headDim,
            Layers = layers,
            ContextDim = (int)weights["transformer.txt_in.weight"].Shape[1].Fixed,
            InChannels = (int)embedder.Shape[1].Fixed,
            // Qwen-Image rotates 16 + 56 + 56 of its 128 head dimensions; other widths keep those proportions.
            AxesDim = [headDim / 8, headDim * 7 / 16, headDim * 7 / 16],
        };
    }

    /// <summary>This configuration with the transformer and scheduler settings of a diffusers folder.</summary>
    /// <param name="schedulerJson">The <c>scheduler_config.json</c> text, or null.</param>
    /// <param name="transformerConfig">The <c>transformer/config.json</c>, or null.</param>
    public QwenImageConfig WithDiffusers(string? schedulerJson, JsonElement? transformerConfig)
    {
        var result = this with { SchedulerJson = schedulerJson };
        if (transformerConfig is { } config && config.TryGetProperty("axes_dims_rope", out _))
        {
            result = result with { AxesDim = config.Ints("axes_dims_rope") };
        }

        return result;
    }

    /// <summary>Writes the transformer, scheduler, family and default keys.</summary>
    /// <param name="metadata">The canonical metadata.</param>
    public void WriteTo(IDictionary<string, object> metadata)
    {
        string t = Prefix + "transformer.";
        int patch = (int)Math.Round(Math.Sqrt(InChannels / (double)LatentChannels));
        metadata[t + "latent_channels"] = LatentChannels;
        metadata[t + "patch_size"] = patch;
        metadata[t + "hidden_size"] = HiddenSize;
        metadata[t + "head_count"] = HeadCount;
        metadata[t + "double_layers"] = Layers;
        metadata[t + "single_layers"] = 0;
        metadata[t + "context_dim"] = ContextDim;
        metadata[t + "pooled_dim"] = 0;
        metadata[t + "axes_dim"] = AxesDim;
        metadata[t + "rope_theta"] = 10000f;
        metadata[t + "centered_positions"] = true;
        metadata[t + "masked_text"] = true;
        metadata[t + "sample_size"] = 1024;

        // The scheduler: the pipeline's σ = linspace(1, 1 / n), the exponential dynamic shift and the terminal stretch.
        float baseShift = 0.5f;
        float maxShift = 0.9f;
        int baseLength = 256;
        int maxLength = 8192;
        float terminal = 0.02f;
        if (SchedulerJson is not null)
        {
            using var document = JsonDocument.Parse(SchedulerJson);
            var root = document.RootElement;
            baseShift = root.Float("base_shift", baseShift);
            maxShift = root.Float("max_shift", maxShift);
            baseLength = root.Int("base_image_seq_len", baseLength);
            maxLength = root.Int("max_image_seq_len", maxLength);
            terminal = root.TryGetProperty("shift_terminal", out var value) && value.ValueKind == JsonValueKind.Number ? value.GetSingle() : 0f;
            metadata[Prefix + "scheduler.source_config"] = SchedulerJson;
        }

        string s = Prefix + "scheduler.";
        metadata[s + "train_steps"] = 1000;
        metadata[s + "shift"] = 1f;
        metadata[s + "dynamic_shift"] = true;
        metadata[s + "base_shift"] = (double)baseShift;
        metadata[s + "max_shift"] = (double)maxShift;
        metadata[s + "base_sequence_length"] = baseLength;
        metadata[s + "max_sequence_length"] = maxLength;
        metadata[s + "shift_terminal"] = (double)terminal;
        metadata[s + "sampler"] = "euler";

        metadata[Prefix + "text_encoder.prompt_length"] = PromptLength;
        metadata[Prefix + "text_encoder.prompt_skip"] = PromptSkip;
        metadata[Prefix + "text_encoder.prompt_template"] = PromptTemplate;
        metadata[Prefix + "default_steps"] = 50;
        metadata[Prefix + "default_guidance"] = 4.0;
    }
}
