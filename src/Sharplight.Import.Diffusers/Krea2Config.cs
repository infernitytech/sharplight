using System.Text.Json;
using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// The configuration of a Krea 2 transformer and its pipeline defaults, written as the canonical <c>krea2.transformer.*</c>,
/// <c>krea2.scheduler.*</c>, text-encoder and default keys: the system-prompt template whose 34-token system turn is dropped from the
/// conditioning, 512 prompt tokens, the Qwen3-VL hidden states entering layers 2, 5, …, 35 stacked per token, the exponential shift (μ from
/// the image token count between 0.5 and 1.15, or the distilled Turbo's fixed 1.15), Turbo: 8 steps without guidance, Raw: 28 steps with
/// guidance 5.5 (diffusers' Krea convention cond + 4.5 · (cond − uncond)).
/// </summary>
internal sealed record Krea2Config
{
    /// <summary>Canonical architecture id.</summary>
    public const string Architecture = "krea2";

    /// <summary>Prefix of the architecture's metadata keys.</summary>
    public const string Prefix = Architecture + ".";

    /// <summary>
    /// The pipeline's prompt template. The pipeline pads the system turn and the prompt and appends the assistant turn after the padding;
    /// the text rows have no positions in the transformer and the padding is masked, so the assistant turn may follow the prompt directly.
    /// </summary>
    public const string PromptTemplate = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of "
        + "the objects and background:<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n";

    /// <summary>Tokens of the system turn the pipeline drops from the hidden states.</summary>
    public const int PromptSkip = 34;

    /// <summary>Prompt tokens after the system turn.</summary>
    public const int PromptLength = 512;

    /// <summary>Model width.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Attention heads.</summary>
    public required int HeadCount { get; init; }

    /// <summary>Key / value heads.</summary>
    public required int KeyValueHeads { get; init; }

    /// <summary>Blocks.</summary>
    public required int Layers { get; init; }

    /// <summary>Text feature width (the text encoder's hidden size).</summary>
    public required int TextDim { get; init; }

    /// <summary>Attention heads of the text-fusion blocks.</summary>
    public required int TextHeads { get; init; }

    /// <summary>Key / value heads of the text-fusion blocks.</summary>
    public required int TextKeyValueHeads { get; init; }

    /// <summary>Text-fusion blocks across the tapped layers.</summary>
    public required int LayerwiseBlocks { get; init; }

    /// <summary>Text-fusion blocks across the tokens.</summary>
    public required int RefinerBlocks { get; init; }

    /// <summary>Patchified latent channels (VAE channels · 2 · 2).</summary>
    public required int InChannels { get; init; }

    /// <summary>
    /// The tapped text-encoder layers (the pipeline's <c>text_encoder_select_layers</c>), or null for transformer-only files: the adapter then
    /// taps every (encoder layers / tapped layers)-th layer, ending one before the last, as the released pipelines (2, 5, …, 35 of 36).
    /// </summary>
    public int[]? TextLayers { get; init; }

    /// <summary>The number of tapped layers the transformer's projector takes.</summary>
    public required int TextLayerCount { get; init; }

    /// <summary>RoPE dimensions per position axis.</summary>
    public int[] AxesDim { get; init; } = [32, 48, 48];

    /// <summary>RoPE base.</summary>
    public float RopeTheta { get; init; } = 1000f;

    /// <summary>Norm epsilon.</summary>
    public float NormEps { get; init; } = 1e-5f;

    /// <summary>Whether the transformer is the few-step distilled Turbo (fixed shift, no guidance).</summary>
    public bool Distilled { get; init; } = true;

    /// <summary>The original scheduler configuration, or null.</summary>
    public string? SchedulerJson { get; init; }

    /// <summary>Infers the configuration from canonical <c>transformer.*</c> weights (128-wide heads as released).</summary>
    /// <param name="weights">The canonical weights.</param>
    /// <exception cref="InvalidDataException">The weights are not a Krea 2 transformer.</exception>
    public static Krea2Config FromWeights(IReadOnlyDictionary<string, WeightInfo> weights)
    {
        if (!weights.TryGetValue("transformer.img_in.weight", out var embedder) || !weights.TryGetValue("transformer.blocks.0.attn.norm_q.weight", out var norm)
            || !weights.TryGetValue("transformer.text_fusion.projector.weight", out var projector))
        {
            throw new InvalidDataException("The checkpoint has no Krea 2 transformer.");
        }

        int hidden = (int)embedder.Shape[0].Fixed;
        int headDim = (int)norm.Shape[0].Fixed;
        int textDim = (int)weights["transformer.txt_in.in_layer.weight"].Shape[1].Fixed;
        int textHeadDim = (int)weights["transformer.text_fusion.refiner.0.attn.norm_q.weight"].Shape[0].Fixed;
        return new Krea2Config
        {
            HiddenSize = hidden,
            HeadCount = hidden / headDim,
            KeyValueHeads = (int)weights["transformer.blocks.0.attn.k.weight"].Shape[0].Fixed / headDim,
            Layers = Count(weights, "transformer.blocks."),
            TextDim = textDim,
            TextHeads = textDim / textHeadDim,
            TextKeyValueHeads = (int)weights["transformer.text_fusion.refiner.0.attn.k.weight"].Shape[0].Fixed / textHeadDim,
            LayerwiseBlocks = Count(weights, "transformer.text_fusion.layerwise."),
            RefinerBlocks = Count(weights, "transformer.text_fusion.refiner."),
            InChannels = (int)embedder.Shape[1].Fixed,
            TextLayerCount = (int)projector.Shape.Elements(),
            AxesDim = [headDim / 4, headDim * 3 / 8, headDim * 3 / 8],
        };
    }

    /// <summary>This configuration with the transformer, scheduler and pipeline settings of a diffusers folder.</summary>
    /// <param name="schedulerJson">The <c>scheduler_config.json</c> text, or null.</param>
    /// <param name="transformerConfig">The <c>transformer/config.json</c>, or null.</param>
    /// <param name="modelIndex">The <c>model_index.json</c>, or null.</param>
    public Krea2Config WithDiffusers(string? schedulerJson, JsonElement? transformerConfig, JsonElement? modelIndex)
    {
        var result = this with { SchedulerJson = schedulerJson };
        if (transformerConfig is { } config)
        {
            result = result with
            {
                AxesDim = config.TryGetProperty("axes_dims_rope", out _) ? config.Ints("axes_dims_rope") : AxesDim,
                RopeTheta = config.Float("rope_theta", RopeTheta),
                NormEps = config.Float("norm_eps", NormEps),
            };
        }

        if (modelIndex is { } index)
        {
            result = result with
            {
                TextLayers = index.TryGetProperty("text_encoder_select_layers", out _) ? index.Ints("text_encoder_select_layers") : TextLayers,
                Distilled = index.Bool("is_distilled", false),
            };
        }

        return result;
    }

    /// <summary>Writes the transformer, scheduler, family and default keys.</summary>
    /// <param name="metadata">The canonical metadata.</param>
    public void WriteTo(IDictionary<string, object> metadata)
    {
        string t = Prefix + "transformer.";
        metadata[t + "in_channels"] = InChannels;
        metadata[t + "hidden_size"] = HiddenSize;
        metadata[t + "head_count"] = HeadCount;
        metadata[t + "key_value_heads"] = KeyValueHeads;
        metadata[t + "layers"] = Layers;
        metadata[t + "text_dim"] = TextDim;
        metadata[t + "text_heads"] = TextHeads;
        metadata[t + "text_key_value_heads"] = TextKeyValueHeads;
        metadata[t + "layerwise_blocks"] = LayerwiseBlocks;
        metadata[t + "refiner_blocks"] = RefinerBlocks;
        metadata[t + "text_layers"] = TextLayerCount;
        metadata[t + "axes_dim"] = AxesDim;
        metadata[t + "rope_theta"] = RopeTheta;
        metadata[t + "norm_eps"] = NormEps;
        metadata[t + "sample_size"] = 1024;
        WriteScheduler(metadata);
        metadata[Prefix + "text_encoder.prompt_length"] = PromptLength;
        metadata[Prefix + "text_encoder.prompt_skip"] = PromptSkip;
        metadata[Prefix + "text_encoder.prompt_template"] = PromptTemplate;
        if (TextLayers is not null)
        {
            metadata[Prefix + "text_encoder.layers"] = TextLayers;
        }

        metadata[Prefix + "default_steps"] = Distilled ? 8 : 28;
        metadata[Prefix + "default_guidance"] = Distilled ? 1.0 : 5.5;
        metadata[Prefix + "classifier_free_guidance"] = !Distilled;
    }

    /// <summary>
    /// The flow scheduler keys: σ = linspace(1, 1 / n) shifted exponentially with μ interpolated from the image token count (the configuration's
    /// base / maximum shift and lengths), or the distilled pipeline's fixed μ = 1.15 (a dynamic shift whose base and maximum are equal).
    /// </summary>
    private void WriteScheduler(IDictionary<string, object> metadata)
    {
        float baseShift = 0.5f;
        float maxShift = 1.15f;
        int baseLength = 256;
        int maxLength = 6400;
        if (SchedulerJson is not null)
        {
            using var document = JsonDocument.Parse(SchedulerJson);
            var root = document.RootElement;
            baseShift = root.Float("base_shift", baseShift);
            maxShift = root.Float("max_shift", maxShift);
            baseLength = root.Int("base_image_seq_len", baseLength);
            maxLength = root.Int("max_image_seq_len", maxLength);
            metadata[Prefix + "scheduler.source_config"] = SchedulerJson;
        }

        string s = Prefix + "scheduler.";
        metadata[s + "train_steps"] = 1000;
        metadata[s + "shift"] = 1f;
        metadata[s + "dynamic_shift"] = true;
        metadata[s + "base_shift"] = Distilled ? 1.15 : baseShift;
        metadata[s + "max_shift"] = Distilled ? 1.15 : maxShift;
        metadata[s + "base_sequence_length"] = baseLength;
        metadata[s + "max_sequence_length"] = maxLength;
        metadata[s + "sampler"] = "euler";
    }

    private static int Count(IReadOnlyDictionary<string, WeightInfo> weights, string prefix)
    {
        int count = 0;
        while (weights.ContainsKey($"{prefix}{count}.attn.norm_q.weight"))
        {
            count++;
        }

        return count;
    }
}
