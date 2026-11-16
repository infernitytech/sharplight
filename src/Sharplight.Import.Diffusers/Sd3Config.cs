using System.Text.Json;
using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// The configuration of an SD3 / SD3.5 transformer and its pipeline defaults, written as the canonical <c>sd3.transformer.*</c>,
/// <c>sd3.scheduler.*</c> and default keys, plus the keys the family fixes for its companions (the CLIP encoders' penultimate-layer
/// conditioning and CLIP-G's <c>!</c> padding, T5 prompts of 256 tokens, the VAE's latent scaling). The shapes come from the canonical
/// weights (every format stores the same tensors); SD3.5 is told apart by its per-head RMS norms and the second attention of its first
/// blocks.
/// </summary>
internal sealed record Sd3Config
{
    /// <summary>Canonical architecture id.</summary>
    public const string Architecture = "sd3";

    /// <summary>Prefix of the architecture's metadata keys.</summary>
    public const string Prefix = Architecture + ".";

    /// <summary>The SD3 VAE's latent scaling.</summary>
    public const float VaeScalingFactor = 1.5305f;

    /// <summary>The SD3 VAE's latent shift.</summary>
    public const float VaeShiftFactor = 0.0609f;

    /// <summary>T5 prompt length of the SD3 pipelines.</summary>
    public const int PromptLength = 256;

    /// <summary>Model width.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Attention heads.</summary>
    public required int HeadCount { get; init; }

    /// <summary>Joint blocks.</summary>
    public required int Layers { get; init; }

    /// <summary>Text conditioning width.</summary>
    public required int ContextDim { get; init; }

    /// <summary>Pooled text embedding width.</summary>
    public required int PooledDim { get; init; }

    /// <summary>Patch input width (latent channels × patch area).</summary>
    public required int InChannels { get; init; }

    /// <summary>Side of the position grid in tokens.</summary>
    public required int PositionGrid { get; init; }

    /// <summary>The first blocks with the second, image-only attention.</summary>
    public required int DualAttentionLayers { get; init; }

    /// <summary>Whether queries and keys are RMS-normed per head.</summary>
    public required bool QkNorm { get; init; }

    /// <summary>Latent channels.</summary>
    public int LatentChannels { get; init; } = 16;

    /// <summary>Flow scheduler: static shift.</summary>
    public float Shift { get; init; } = 3f;

    /// <summary>The original scheduler configuration, or null.</summary>
    public string? SchedulerJson { get; init; }

    /// <summary>Infers the configuration from canonical <c>transformer.*</c> weights, with the family's defaults.</summary>
    /// <param name="weights">The canonical weights.</param>
    /// <exception cref="InvalidDataException">The weights are not an SD3 transformer.</exception>
    public static Sd3Config FromWeights(IReadOnlyDictionary<string, WeightInfo> weights)
    {
        if (!weights.TryGetValue("transformer.patch_embed.weight", out var patch) || !weights.TryGetValue("transformer.pos_embed", out var positions))
        {
            throw new InvalidDataException("The checkpoint has no SD3 transformer.");
        }

        int hidden = (int)patch.Shape[0].Fixed;
        // Files without q/k norms (SD3) do not tell the head width: every SD3 release has 64-wide heads.
        int headDim = weights.TryGetValue("transformer.joint.0.img_attn.norm_q.weight", out var norm) ? (int)norm.Shape[0].Fixed : 64;
        int dual = 0;
        while (weights.ContainsKey($"transformer.joint.{dual}.img_attn2.qkv.weight"))
        {
            dual++;
        }

        return new Sd3Config
        {
            HiddenSize = hidden,
            HeadCount = hidden / headDim,
            Layers = Sd3Names.CountBlocks(weights),
            ContextDim = (int)weights["transformer.txt_in.weight"].Shape[1].Fixed,
            PooledDim = (int)weights["transformer.vector_in.in_layer.weight"].Shape[1].Fixed,
            InChannels = (int)patch.Shape[1].Fixed,
            PositionGrid = (int)Math.Round(Math.Sqrt(positions.Shape[0].Fixed)),
            DualAttentionLayers = dual,
            QkNorm = norm is not null,
        };
    }

    /// <summary>This configuration with the scheduler settings and the head width of a diffusers folder.</summary>
    /// <param name="schedulerJson">The <c>scheduler_config.json</c> text, or null.</param>
    /// <param name="transformerConfig">The <c>transformer/config.json</c>, or null.</param>
    public Sd3Config WithDiffusers(string? schedulerJson, JsonElement? transformerConfig)
    {
        var result = this;
        if (transformerConfig is { } config && config.TryGetProperty("attention_head_dim", out var headDim))
        {
            result = result with { HeadCount = HiddenSize / headDim.GetInt32() };
        }

        return result.WithSchedulerJson(schedulerJson);
    }

    private Sd3Config WithSchedulerJson(string? schedulerJson)
    {
        if (schedulerJson is null)
        {
            return this;
        }

        using var document = JsonDocument.Parse(schedulerJson);
        return this with { Shift = document.RootElement.Float("shift", 3f), SchedulerJson = schedulerJson };
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
        metadata[t + "layers"] = Layers;
        metadata[t + "context_dim"] = ContextDim;
        metadata[t + "pooled_dim"] = PooledDim;
        metadata[t + "position_grid"] = PositionGrid;
        metadata[t + "dual_attention_layers"] = DualAttentionLayers;
        metadata[t + "qk_norm"] = QkNorm;
        // 1024 pixels, unless the position grid covers less (tiny test models).
        metadata[t + "sample_size"] = Math.Min(1024, PositionGrid * patch * 8);
        string s = Prefix + "scheduler.";
        metadata[s + "train_steps"] = 1000;
        metadata[s + "shift"] = Shift;
        metadata[s + "dynamic_shift"] = false;
        metadata[s + "sampler"] = "euler";
        // The SD3 pipelines let the scheduler space the noise levels over its training range.
        metadata[s + "timestep_spacing"] = "scheduler";
        if (SchedulerJson is not null)
        {
            metadata[s + "source_config"] = SchedulerJson;
        }

        // SD3 conditions on both CLIP encoders' penultimate layers without the final norm; CLIP-G prompts are padded with "!" (id 0).
        foreach (string component in new[] { "text_encoder", "text_encoder_2" })
        {
            metadata[$"{Prefix}{component}.clip_skip"] = 2;
            metadata[$"{Prefix}{component}.final_norm"] = false;
        }

        metadata[Prefix + "text_encoder_2.pad_id"] = 0;
        metadata[Prefix + "text_encoder_3.prompt_length"] = PromptLength;
        metadata[Prefix + "vae.scaling_factor"] = VaeScalingFactor;
        metadata[Prefix + "vae.shift_factor"] = VaeShiftFactor;
        metadata[Prefix + "default_steps"] = 28;
        metadata[Prefix + "default_guidance"] = 7.0;
    }
}
