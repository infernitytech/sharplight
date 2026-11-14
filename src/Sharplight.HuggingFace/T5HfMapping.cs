using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>t5</c>, <c>mt5</c> and <c>umt5</c> (T5ForConditionalGeneration, T5EncoderModel) → canonical <c>encoder_decoder</c>
/// (vocabulary documented on <c>EncoderDecoderAdapter</c>). The bucketed relative position bias of the first block serves its stack
/// (UMT5's per-block tables stay per layer) and is stored as f32; the LM head is <c>lm_head.weight</c> when the checkpoint has one,
/// else the shared embedding, and the decoder output is scaled by d_model^-0.5 when the config says so (<c>scale_decoder_outputs</c>,
/// older configs: tied word embeddings, the original T5).
/// </summary>
internal sealed partial class T5HfMapping : IHfArchitectureMapping
{
    /// <inheritdoc/>
    public string Architecture => "encoder_decoder";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["t5", "mt5", "umt5"];

    private static bool EncoderOnly(HfConfig c) => c.Architectures.Any(a => a.EndsWith("EncoderModel", StringComparison.Ordinal));

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HfImportOptions options)
    {
        string a = Architecture + ".";
        int layers = c.Int("num_layers");
        string proj = c.GetString("feed_forward_proj") ?? "relu";
        string act = proj.Split('-')[^1];
        if (proj == "gated-gelu")
        {
            act = "gelu_new";   // transformers' backwards-compatible meaning of "gated-gelu"
        }

        bool scaled = c.GetBool("scale_decoder_outputs") ?? c.GetBool("tie_word_embeddings") ?? true;
        int hidden = c.Int("d_model");
        return new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "hidden_size"] = hidden,
            [a + "encoder.layer_count"] = layers,
            [a + "decoder.layer_count"] = EncoderOnly(c) ? 0 : c.GetInt("num_decoder_layers") ?? layers,
            [a + "attention.head_count"] = c.Int("num_heads"),
            [a + "attention.head_dim"] = c.Int("d_kv"),
            [a + "norm"] = "rms",
            [a + "norm_eps"] = (float)(c.GetDouble("layer_norm_epsilon") ?? 1e-6),
            [a + "mlp.activation"] = act switch
            {
                "relu" => "relu",
                "gelu" => "gelu",
                "gelu_new" or "gelu_pytorch_tanh" or "gelu_fast" => "gelu_tanh",
                "silu" or "swish" => "silu",
                _ => throw new NotSupportedException($"{c.ModelType}: feed_forward_proj '{proj}' is not supported."),
            },
            [a + "relative_attention.max_distance"] = c.GetInt("relative_attention_max_distance") ?? 128,
            [a + "decoder_start_token"] = c.GetInt("decoder_start_token_id") ?? c.GetInt("pad_token_id") ?? 0,
            [a + "output.scale"] = scaled ? 1f / MathF.Sqrt(hidden) : 1f,
            [a + "context_length"] = c.GetInt("n_positions") ?? 512,
        };
    }

    [GeneratedRegex(@"^(encoder|decoder)\.block\.(\d+)\.layer\.(\d+)\.(.+)$")]
    private static partial Regex BlockName();

    /// <summary>Sub-layer modules by stack and sub-layer index (self-attention, [cross-attention,] feed-forward).</summary>
    private static readonly Dictionary<string, string> s_modules = new(StringComparer.Ordinal)
    {
        ["0.SelfAttention.q"] = "attention.q",
        ["0.SelfAttention.k"] = "attention.k",
        ["0.SelfAttention.v"] = "attention.v",
        ["0.SelfAttention.o"] = "attention.o",
        ["0.layer_norm"] = "attention_norm",
        ["1.EncDecAttention.q"] = "cross_attention.q",
        ["1.EncDecAttention.k"] = "cross_attention.k",
        ["1.EncDecAttention.v"] = "cross_attention.v",
        ["1.EncDecAttention.o"] = "cross_attention.o",
        ["1.layer_norm"] = "cross_attention_norm",
    };

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HfImportOptions options)
    {
        var collector = new HfWeightCollector();
        foreach (var (name, tensor) in tensors)
        {
            var info = tensor.ToWeightInfo();
            switch (name)
            {
                case "shared.weight":
                    collector.Add(info.As("token_embedding.weight"));
                    continue;
                case "encoder.embed_tokens.weight" or "decoder.embed_tokens.weight":
                    continue;   // the shared embedding, saved again by older checkpoints
                case "lm_head.weight":
                    if (!EncoderOnly(c))
                    {
                        collector.Add(info.As("output.weight"));
                    }

                    continue;
                case "encoder.final_layer_norm.weight":
                    collector.Add(HfTransforms.Plain(info, "encoder.output_norm.weight"));
                    continue;
                case "decoder.final_layer_norm.weight":
                    collector.Add(HfTransforms.Plain(info, "decoder.output_norm.weight"));
                    continue;
            }

            var m = BlockName().Match(name);
            if (!m.Success)
            {
                throw new NotSupportedException($"{c.ModelType}: tensor '{name}' has no canonical mapping.");
            }

            string stack = m.Groups[1].Value;
            string layer = m.Groups[2].Value;
            int sub = int.Parse(m.Groups[3].Value, System.Globalization.CultureInfo.InvariantCulture);
            string rest = m.Groups[4].Value;
            if (rest == "SelfAttention.relative_attention_bias.weight")
            {
                string target = layer == "0" ? $"{stack}.attention.relative_bias.weight" : $"{stack}.layers.{layer}.attention.relative_bias.weight";
                collector.Add(HfTransforms.ToF32(info, target));
                continue;
            }

            // The feed-forward sub-layer is the last one: 1 in the encoder, 2 in the decoder.
            int ffn = stack == "encoder" ? 1 : 2;
            string? module = sub == ffn
                ? rest switch
                {
                    "layer_norm.weight" => "mlp_norm.weight",
                    "DenseReluDense.wi.weight" or "DenseReluDense.wi_1.weight" => "mlp.up.weight",
                    "DenseReluDense.wi_0.weight" => "mlp.gate.weight",
                    "DenseReluDense.wo.weight" => "mlp.down.weight",
                    _ => null,
                }
                : rest.EndsWith(".weight", StringComparison.Ordinal) && s_modules.TryGetValue($"{sub}.{rest[..^".weight".Length]}", out var mapped)
                    && (sub == 0 || stack == "decoder") ? mapped + ".weight" : null;
            if (module is null)
            {
                throw new NotSupportedException($"{c.ModelType}: tensor '{name}' has no canonical mapping.");
            }

            string canonical = $"{stack}.layers.{layer}.{module}";
            collector.Add(module.EndsWith("norm.weight", StringComparison.Ordinal) ? HfTransforms.Plain(info, canonical) : info.As(canonical));
        }

        return collector.Weights;
    }
}
