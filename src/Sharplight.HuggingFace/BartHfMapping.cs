using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face BART-family encoder-decoders (<c>bart</c>: post-norm; <c>mbart</c>: pre-norm with final layer norms and embeddings scaled by
/// √d_model when <c>scale_embedding</c>) → canonical <c>encoder_decoder</c> (vocabulary documented on <c>EncoderDecoderAdapter</c>). The learned
/// position tables lose BART's first two rows (position p is stored at row p + 2), <c>final_logits_bias</c> becomes the LM head's bias and
/// the attention scale is head_dim^-0.5 (BART scales its query projection).
/// </summary>
internal sealed partial class BartHfMapping : IHfArchitectureMapping
{
    private const int PositionOffset = 2;

    /// <inheritdoc/>
    public string Architecture => "encoder_decoder";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["bart", "mbart"];

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HfImportOptions options)
    {
        string a = Architecture + ".";
        int hidden = c.Int("d_model");
        int heads = c.Int("encoder_attention_heads");
        if (c.GetInt("decoder_attention_heads") is { } decoderHeads && decoderHeads != heads)
        {
            throw new NotSupportedException($"{c.ModelType}: different encoder and decoder head counts are not supported.");
        }

        string act = c.GetString("activation_function") ?? "gelu";
        return new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "hidden_size"] = hidden,
            [a + "encoder.layer_count"] = c.Int("encoder_layers"),
            [a + "decoder.layer_count"] = c.Int("decoder_layers"),
            [a + "attention.head_count"] = heads,
            [a + "attention.head_dim"] = hidden / heads,
            [a + "attention.scale"] = 1f / MathF.Sqrt(hidden / heads),
            [a + "norm"] = "layer",
            [a + "norm_eps"] = 1e-5f,
            [a + "pre_norm"] = c.ModelType == "mbart",
            [a + "embedding_scale"] = c.GetBool("scale_embedding") == true ? MathF.Sqrt(hidden) : 1f,
            [a + "mlp.activation"] = act switch
            {
                "gelu" => "gelu",
                "gelu_new" or "gelu_pytorch_tanh" or "gelu_fast" => "gelu_tanh",
                "relu" => "relu",
                "silu" or "swish" => "silu",
                _ => throw new NotSupportedException($"{c.ModelType}: activation '{act}' is not supported."),
            },
            [a + "decoder_start_token"] = c.GetInt("decoder_start_token_id") ?? c.GetInt("eos_token_id") ?? 2,
            [a + "context_length"] = c.GetInt("max_position_embeddings") ?? 1024,
        };
    }

    [GeneratedRegex(@"^(encoder|decoder)\.layers\.(\d+)\.(.+)\.(weight|bias)$")]
    private static partial Regex LayerName();

    /// <summary>Per-layer modules, HF name to canonical name.</summary>
    private static readonly Dictionary<string, string> s_layer = new(StringComparer.Ordinal)
    {
        ["self_attn.q_proj"] = "attention.q",
        ["self_attn.k_proj"] = "attention.k",
        ["self_attn.v_proj"] = "attention.v",
        ["self_attn.out_proj"] = "attention.o",
        ["self_attn_layer_norm"] = "attention_norm",
        ["encoder_attn.q_proj"] = "cross_attention.q",
        ["encoder_attn.k_proj"] = "cross_attention.k",
        ["encoder_attn.v_proj"] = "cross_attention.v",
        ["encoder_attn.out_proj"] = "cross_attention.o",
        ["encoder_attn_layer_norm"] = "cross_attention_norm",
        ["fc1"] = "mlp.up",
        ["fc2"] = "mlp.down",
        ["final_layer_norm"] = "mlp_norm",
    };

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HfImportOptions options)
    {
        var collector = new HfWeightCollector();
        foreach (var (rawName, tensor) in tensors)
        {
            var info = tensor.ToWeightInfo();
            string name = rawName.StartsWith("model.", StringComparison.Ordinal) ? rawName["model.".Length..] : rawName;
            switch (name)
            {
                case "shared.weight":
                    collector.Add(info.As("token_embedding.weight"));
                    continue;
                case "encoder.embed_tokens.weight" or "decoder.embed_tokens.weight":
                    continue;   // the shared embedding, saved again by some checkpoints
                case "lm_head.weight":
                    collector.Add(info.As("output.weight"));
                    continue;
                case "final_logits_bias":
                    collector.Add(HfTransforms.ToF32(info, "output.bias", new Shape(info.Shape.Elements())));
                    continue;
                case "encoder.embed_positions.weight" or "decoder.embed_positions.weight":
                    string stack = name[..name.IndexOf('.', StringComparison.Ordinal)];
                    collector.Add(info.RowSlice($"{stack}.position_embedding.weight", PositionOffset, info.Shape[0].Fixed - PositionOffset));
                    continue;
                case "encoder.layernorm_embedding.weight" or "encoder.layernorm_embedding.bias" or "decoder.layernorm_embedding.weight" or "decoder.layernorm_embedding.bias":
                    collector.Add(HfTransforms.Plain(info, name.Replace("layernorm_embedding", "embedding_norm", StringComparison.Ordinal)));
                    continue;
                case "encoder.layer_norm.weight" or "encoder.layer_norm.bias" or "decoder.layer_norm.weight" or "decoder.layer_norm.bias":
                    collector.Add(HfTransforms.Plain(info, name.Replace("layer_norm", "output_norm", StringComparison.Ordinal)));
                    continue;
            }

            var m = LayerName().Match(name);
            if (!m.Success || !s_layer.TryGetValue(m.Groups[3].Value, out var module))
            {
                throw new NotSupportedException($"{c.ModelType}: tensor '{rawName}' has no canonical mapping.");
            }

            string canonical = $"{m.Groups[1].Value}.layers.{m.Groups[2].Value}.{module}.{m.Groups[4].Value}";
            collector.Add(info.Shape.Rank == 1 ? HfTransforms.Plain(info, canonical) : info.As(canonical));
        }

        return collector.Weights;
    }
}
