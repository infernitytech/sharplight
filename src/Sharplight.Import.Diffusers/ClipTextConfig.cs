using System.Text.Json;
using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>Hyper-parameters of a CLIP text encoder, written as the canonical <c>latent_diffusion.{component}.*</c> keys.</summary>
internal sealed record ClipTextConfig
{
    /// <summary>Transformer layers.</summary>
    public required int LayerCount { get; init; }

    /// <summary>Model width.</summary>
    public required int HiddenSize { get; init; }

    /// <summary>Attention heads.</summary>
    public required int HeadCount { get; init; }

    /// <summary>MLP width.</summary>
    public required int FfnHiddenSize { get; init; }

    /// <summary>MLP activation: "quick_gelu" (OpenAI CLIP) or "gelu" (OpenCLIP).</summary>
    public required string Activation { get; init; }

    /// <summary>Tokens per sequence (77).</summary>
    public int ContextLength { get; init; } = 77;

    /// <summary>LayerNorm epsilon.</summary>
    public float NormEps { get; init; } = 1e-5f;

    /// <summary>
    /// How many layers from the end the conditioning is taken (1: the last layer; 2: the penultimate one, the SD 2.x / SDXL
    /// default).
    /// </summary>
    public int ClipSkip { get; init; } = 1;

    /// <summary>Whether the final LayerNorm is applied to the conditioning (SD 1.x / 2.x: yes; SDXL: no).</summary>
    public bool FinalNorm { get; init; } = true;

    /// <summary>Writes the keys under <c>{prefix}{component}.</c>.</summary>
    /// <param name="metadata">The canonical metadata.</param>
    /// <param name="prefix">The architecture prefix (<c>latent_diffusion.</c>).</param>
    /// <param name="component">The encoder's key (<c>text_encoder</c>).</param>
    public void WriteTo(IDictionary<string, object> metadata, string prefix, string component)
    {
        string p = $"{prefix}{component}.";
        metadata[p + "layer_count"] = LayerCount;
        metadata[p + "hidden_size"] = HiddenSize;
        metadata[p + "head_count"] = HeadCount;
        metadata[p + "ffn_hidden_size"] = FfnHiddenSize;
        metadata[p + "activation"] = Activation;
        metadata[p + "context_length"] = ContextLength;
        metadata[p + "norm_eps"] = NormEps;
        metadata[p + "clip_skip"] = ClipSkip;
        metadata[p + "final_norm"] = FinalNorm;
    }

    /// <summary>Reads a transformers <c>text_encoder/config.json</c> (<c>CLIPTextConfig</c>).</summary>
    /// <param name="config">The parsed configuration.</param>
    public static ClipTextConfig FromTransformers(JsonElement config)
    {
        var text = config.TryGetProperty("text_config", out var nested) ? nested : config;
        return new ClipTextConfig
        {
            LayerCount = text.Int("num_hidden_layers", 12),
            HiddenSize = text.Int("hidden_size", 768),
            HeadCount = text.Int("num_attention_heads", 12),
            FfnHiddenSize = text.Int("intermediate_size", 3072),
            Activation = text.String("hidden_act", "quick_gelu"),
            ContextLength = text.Int("max_position_embeddings", 77),
            NormEps = text.Float("layer_norm_eps", 1e-5f),
        };
    }

    /// <summary>Infers the configuration from the canonical weights under <paramref name="prefix"/> (64-wide heads, as in every CLIP).</summary>
    /// <param name="weights">The canonical weights.</param>
    /// <param name="prefix">Canonical prefix of the encoder (<c>text_encoder.</c>).</param>
    /// <param name="activation">The MLP activation of the family.</param>
    public static ClipTextConfig FromWeights(IReadOnlyDictionary<string, WeightInfo> weights, string prefix, string activation)
    {
        int layers = 0;
        while (weights.ContainsKey($"{prefix}layers.{layers}.norm1.weight"))
        {
            layers++;
        }

        int hidden = (int)weights[prefix + "token_embedding.weight"].Shape[1].Fixed;
        return new ClipTextConfig
        {
            LayerCount = layers,
            HiddenSize = hidden,
            HeadCount = hidden / 64,
            FfnHiddenSize = (int)weights[prefix + "layers.0.mlp.up.weight"].Shape[0].Fixed,
            Activation = activation,
            ContextLength = (int)weights[prefix + "position_embedding.weight"].Shape[0].Fixed,
        };
    }
}
