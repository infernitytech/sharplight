using System.Text.Json;
using Sharplight.Import.Gguf;

namespace Sharplight.Studio.Services.Hub;

/// <summary>Turns what can be read of a model before downloading it into a <see cref="ModelShape"/>.</summary>
internal static class ModelShapeReader
{
    /// <summary>Configuration objects that hold the language model's settings in multimodal checkpoints.</summary>
    private static readonly string[] s_nested = ["text_config", "llm_config", "language_config"];

    /// <summary>The shape stated by the start of a GGUF file (null when it holds none of it).</summary>
    /// <param name="prefix">The first bytes of the file.</param>
    public static ModelShape? FromGgufPrefix(ReadOnlySpan<byte> prefix) => FromGguf(GgufHeaderProbe.ReadShape(prefix));

    /// <summary>The shape of GGUF dimensions (null when the layer count is missing).</summary>
    /// <param name="shape">The dimensions.</param>
    public static ModelShape? FromGguf(GgufModelShape shape)
    {
        if (shape.BlockCount is null)
        {
            return null;
        }

        return new ModelShape(shape.BlockCount, shape.HeadCount, shape.KvHeadCount, shape.EmbeddingLength, shape.KeyLength, shape.ValueLength,
            shape.ContextLength, shape.VocabularySize, shape.KvLoraRank, shape.KvLoraRank is > 0 ? shape.RopeDimension : null);
    }

    /// <summary>The shape a Hugging Face <c>config.json</c> states (null when it is not one or has no layer count).</summary>
    /// <param name="json">The file's text.</param>
    public static ModelShape? FromConfig(string json)
    {
        try
        {
            using var document = JsonDocument.Parse(json);
            var root = document.RootElement;
            if (root.ValueKind != JsonValueKind.Object)
            {
                return null;
            }

            foreach (var name in s_nested)
            {
                if (root.TryGetProperty(name, out var nested) && nested.ValueKind == JsonValueKind.Object && Int(nested, "num_hidden_layers") is not null)
                {
                    root = nested;
                    break;
                }
            }

            int? layers = Int(root, "num_hidden_layers", "n_layer", "num_layers");
            if (layers is null)
            {
                return null;
            }

            int? heads = Int(root, "num_attention_heads", "n_head");
            int? latent = Int(root, "kv_lora_rank");
            return new ModelShape(layers, heads, Int(root, "num_key_value_heads", "num_kv_heads") ?? heads, Int(root, "hidden_size", "n_embd", "d_model"),
                Int(root, "head_dim"), null, Int(root, "max_position_embeddings", "n_positions"), Int(root, "vocab_size"), latent,
                latent is > 0 ? Int(root, "qk_rope_head_dim") : null);
        }
        catch (JsonException)
        {
            return null;
        }
    }

    private static int? Int(JsonElement element, params string[] names)
    {
        foreach (var name in names)
        {
            if (element.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.Number && value.TryGetInt32(out int number) && number > 0)
            {
                return number;
            }
        }

        return null;
    }
}
