using System.Text.Json;
using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// The configuration of a Qwen-Image 2.1 transformer and its pipeline defaults, written as the canonical <c>qwen_image21.transformer.*</c>,
/// <c>qwen_image21.scheduler.*</c>, text-encoder and default keys (the system-prompt template whose system turn is dropped from the
/// conditioning, the last hidden states before the final norm, 40 steps without guidance, the exponential dynamic shift with the terminal
/// stretch).
/// </summary>
internal sealed record QwenImage21Config
{
    /// <summary>Canonical architecture id.</summary>
    public const string Architecture = "qwen_image21";

    /// <summary>Prefix of the architecture's metadata keys.</summary>
    public const string Prefix = Architecture + ".";

    /// <summary>The pipeline's text-to-image prompt template.</summary>
    public const string PromptTemplate = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n"
        + "<|im_start|>assistant\n";

    /// <summary>Tokens of the system turn the pipeline drops from the hidden states.</summary>
    public const int PromptSkip = 14;

    /// <summary>Prompt tokens after the system turn.</summary>
    public const int PromptLength = 512;

    /// <summary>Model width.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Attention heads.</summary>
    public required int HeadCount { get; init; }

    /// <summary>Blocks.</summary>
    public required int Layers { get; init; }

    /// <summary>Text feature width.</summary>
    public required int ContextDim { get; init; }

    /// <summary>Latent channels.</summary>
    public required int LatentChannels { get; init; }

    /// <summary>RoPE dimensions per position axis.</summary>
    public int[] AxesDim { get; init; } = [16, 56, 56];

    /// <summary>Whether text tokens are modulated from t = 0.</summary>
    public bool CausalCondition { get; init; } = true;

    /// <summary>
    /// The 8-step noise levels of Qwen-Image 2.1 Turbo (its pipeline's <c>sample_sigmas</c>), for transformer-only files recognized by their
    /// published name.
    /// </summary>
    public static readonly float[] TurboSigmas = [1.0f, 0.978453f, 0.95418f, 0.926626f, 0.89508f, 0.845148f, 0.704534f, 0.414568f];

    /// <summary>The pipeline's sampling noise levels before the shift (<c>sample_sigmas</c>, Turbo), or null.</summary>
    public float[]? SampleSigmas { get; init; }

    /// <summary>The original scheduler configuration, or null.</summary>
    public string? SchedulerJson { get; init; }

    /// <summary>Infers the configuration from canonical <c>transformer.*</c> weights (128-wide heads, as the release).</summary>
    /// <param name="weights">The canonical weights.</param>
    /// <exception cref="InvalidDataException">The weights are not a Qwen-Image 2.1 transformer.</exception>
    public static QwenImage21Config FromWeights(IReadOnlyDictionary<string, WeightInfo> weights)
    {
        if (!weights.TryGetValue("transformer.img_in.weight", out var embedder) || !weights.TryGetValue("transformer.blocks.0.attn.norm_q.weight", out var norm))
        {
            throw new InvalidDataException("The checkpoint has no Qwen-Image 2.1 transformer.");
        }

        int hidden = (int)embedder.Shape[0].Fixed;
        int headDim = (int)norm.Shape[0].Fixed;
        int layers = 0;
        while (weights.ContainsKey($"transformer.blocks.{layers}.attn.norm_q.weight"))
        {
            layers++;
        }

        return new QwenImage21Config
        {
            HiddenSize = hidden,
            HeadCount = hidden / headDim,
            Layers = layers,
            ContextDim = (int)weights["transformer.txt_in.in_layer.weight"].Shape[1].Fixed,
            LatentChannels = (int)embedder.Shape[1].Fixed,
            AxesDim = [headDim / 8, headDim * 7 / 16, headDim * 7 / 16],
        };
    }

    /// <summary>This configuration with the transformer and scheduler settings of a diffusers folder.</summary>
    /// <param name="schedulerJson">The <c>scheduler_config.json</c> text, or null.</param>
    /// <param name="transformerConfig">The <c>transformer/config.json</c>, or null.</param>
    public QwenImage21Config WithDiffusers(string? schedulerJson, JsonElement? transformerConfig)
    {
        var result = this with { SchedulerJson = schedulerJson };
        if (transformerConfig is { } config)
        {
            result = result with
            {
                AxesDim = config.TryGetProperty("axes_dims_rope", out _) ? config.Ints("axes_dims_rope") : AxesDim,
                CausalCondition = config.Bool("causal_condition", true),
            };
        }

        return result;
    }

    /// <summary>Writes the transformer, scheduler, family and default keys.</summary>
    /// <param name="metadata">The canonical metadata.</param>
    public void WriteTo(IDictionary<string, object> metadata)
    {
        string t = Prefix + "transformer.";
        metadata[t + "latent_channels"] = LatentChannels;
        metadata[t + "hidden_size"] = HiddenSize;
        metadata[t + "head_count"] = HeadCount;
        metadata[t + "layers"] = Layers;
        metadata[t + "context_dim"] = ContextDim;
        metadata[t + "axes_dim"] = AxesDim;
        metadata[t + "rope_theta"] = 10000f;
        metadata[t + "norm_eps"] = 1e-6f;
        metadata[t + "sample_size"] = 1024;
        metadata[t + "causal_condition"] = CausalCondition;
        QwenImageConfig.WriteScheduler(metadata, Prefix, SchedulerJson);
        metadata[Prefix + "text_encoder.prompt_length"] = PromptLength;
        metadata[Prefix + "text_encoder.prompt_skip"] = PromptSkip;
        metadata[Prefix + "text_encoder.prompt_template"] = PromptTemplate;
        if (SampleSigmas is { Length: > 0 } sigmas)
        {
            metadata[Prefix + "scheduler.sample_sigmas"] = sigmas;
        }

        metadata[Prefix + "default_steps"] = SampleSigmas is { Length: > 0 } turbo ? turbo.Length : 40;
        metadata[Prefix + "default_guidance"] = 1.0;
    }
}
