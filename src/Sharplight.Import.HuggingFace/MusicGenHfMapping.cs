using System.Text.Json;
using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Hugging Face MusicGen (<c>musicgen</c>: MusicgenForConditionalGeneration, e.g. facebook/musicgen-small) → canonical <c>musicgen</c> (documented on
/// <c>MusicGenAdapter</c>): the T5 text encoder, the projection into the decoder, the decoder with its codebook embeddings, sinusoidal position table and heads, and the
/// decoder half of the 32 kHz EnCodec audio codec (weight normalization folded, as in <c>EncodecHfMapping</c>; the codec's encoder is not needed to generate and is dropped).
/// </summary>
internal sealed partial class MusicGenHfMapping : IHfArchitectureMapping
{
    /// <inheritdoc/>
    public string Architecture => "musicgen";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["musicgen"];

    /// <inheritdoc/>
    public IEnumerable<string> Validate(HfConfig c)
    {
        var text = Sub(c, "text_encoder");
        var decoder = Sub(c, "decoder");
        var codec = Sub(c, "audio_encoder");
        if (Str(text, "feed_forward_proj", "relu") != "relu")
        {
            yield return "musicgen: only a T5 text encoder with a ReLU feed-forward is supported.";
        }

        if (Str(decoder, "activation_function", "gelu") != "gelu")
        {
            yield return "musicgen: only a GELU decoder is supported.";
        }

        if (Num(decoder, "audio_channels", 1) != 1)
        {
            yield return "musicgen: only the mono models are supported (the stereo models interleave two codebook sets).";
        }

        if (Str(codec, "model_type", "encodec") != "encodec" || Str(codec, "norm_type", "weight_norm") != "weight_norm")
        {
            yield return "musicgen: only a weight-normalized EnCodec audio codec is supported.";
        }

        if (Bool(decoder, "scale_embedding", false))
        {
            yield return "musicgen: scaled embeddings are not supported.";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HuggingFaceImportOptions options)
    {
        var text = Sub(c, "text_encoder");
        var decoder = Sub(c, "decoder");
        var codec = Sub(c, "audio_encoder");
        string a = Architecture + ".";
        long[] ratios = [.. codec.GetProperty("upsampling_ratios").EnumerateArray().Select(v => v.GetInt64())];
        int sampleRate = (int)Num(codec, "sampling_rate", 32000);
        int codebooks = (int)Num(decoder, "num_codebooks", 4);
        return new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "layer_count"] = (int)Num(decoder, "num_hidden_layers", 24),
            [a + "hidden_size"] = (int)Num(decoder, "hidden_size", 1024),
            [a + "attention.head_count"] = (int)Num(decoder, "num_attention_heads", 16),
            [a + "codebook_count"] = codebooks,
            [a + "vocabulary"] = (int)Num(decoder, "vocab_size", 2048),
            [a + "max_steps"] = (int)Num(decoder, "max_position_embeddings", 2048),
            [a + "norm_eps"] = 1e-5f,
            [a + "guidance_scale"] = 3f,
            [a + "frame_rate"] = sampleRate / (double)ratios.Aggregate(1L, (x, y) => x * y),
            [a + "text.layer_count"] = (int)Num(text, "num_layers", 12),
            [a + "text.hidden_size"] = (int)Num(text, "d_model", 768),
            [a + "text.attention.head_count"] = (int)Num(text, "num_heads", 12),
            [a + "text.head_dim"] = (int)Num(text, "d_kv", 64),
            [a + "text.norm_eps"] = (float)Num(text, "layer_norm_epsilon", 1e-6),
            [a + "text.relative_buckets"] = (int)Num(text, "relative_attention_num_buckets", 32),
            [a + "text.relative_max_distance"] = (int)Num(text, "relative_attention_max_distance", 128),
            [a + "text.context_length"] = 512,
            [a + "codec.sample_rate"] = sampleRate,
            [a + "codec.hidden_size"] = (int)Num(codec, "hidden_size", 128),
            [a + "codec.num_filters"] = (int)Num(codec, "num_filters", 64),
            [a + "codec.kernel_size"] = (int)Num(codec, "kernel_size", 7),
            [a + "codec.last_kernel_size"] = (int)Num(codec, "last_kernel_size", 7),
            [a + "codec.residual_kernel_size"] = (int)Num(codec, "residual_kernel_size", 3),
            [a + "codec.dilation_growth_rate"] = (int)Num(codec, "dilation_growth_rate", 2),
            [a + "codec.residual_layers"] = (int)Num(codec, "num_residual_layers", 1),
            [a + "codec.compress"] = (int)Num(codec, "compress", 2),
            [a + "codec.lstm_layers"] = (int)Num(codec, "num_lstm_layers", 2),
            [a + "codec.upsampling_ratios"] = ratios,
            [a + "codec.codebook_size"] = (int)Num(codec, "codebook_size", 2048),
            [a + "codec.codebook_dim"] = (int)Num(codec, "codebook_dim", 128),
            [a + "codec.codebook_count"] = codebooks,
            [a + "codec.conv_shortcut"] = Bool(codec, "use_conv_shortcut", true),
            [a + "codec.causal"] = Bool(codec, "use_causal_conv", true),
        };
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
    {
        int codebooks = (int)Num(Sub(c, "decoder"), "num_codebooks", 4);
        var collector = new HfWeightCollector();
        foreach (var (name, tensor) in tensors)
        {
            var info = tensor.ToWeightInfo();
            if (name.StartsWith("audio_encoder.", StringComparison.Ordinal))
            {
                MapCodec(collector, tensors, name, tensor, codebooks);
                continue;
            }

            string? mapped = Map(name);
            if (mapped is null)
            {
                collector.AddUnmapped(name);
                continue;
            }

            collector.Add(mapped is "decoder.position_table.weight" or "text.relative_bias.weight" ? HfTransforms.ToF32(info, mapped) : info.Shape.Rank <= 1 ? HfTransforms.Plain(info, mapped) : info.As(mapped));
        }

        collector.ThrowIfUnmapped(c.ModelType);
        return collector.Weights;
    }

    private static void MapCodec(HfWeightCollector collector, IReadOnlyDictionary<string, SafetensorsTensor> tensors, string name, SafetensorsTensor tensor, int codebooks)
    {
        string rest = name["audio_encoder.".Length..];
        if (rest.StartsWith("encoder.", StringComparison.Ordinal) || rest.EndsWith(".cluster_size", StringComparison.Ordinal) || rest.EndsWith(".embed_avg", StringComparison.Ordinal)
            || rest.EndsWith(".inited", StringComparison.Ordinal))
        {
            return;
        }

        if (WeightNorm().Match(rest) is { Success: true } norm)
        {
            if (norm.Groups[2].Value == "g")
            {
                string stem = norm.Groups[1].Value;
                collector.Add(HfWeightNormFolding.Fold(tensor.ToWeightInfo(), tensors["audio_encoder." + stem + ".conv.weight_v"].ToWeightInfo(), "codec." + stem + ".weight"));
            }

            return;
        }

        if (rest.EndsWith(".conv.bias", StringComparison.Ordinal))
        {
            collector.Add(HfTransforms.Plain(tensor.ToWeightInfo(), "codec." + rest.Replace(".conv.bias", ".bias", StringComparison.Ordinal)));
        }
        else if (LstmInput().Match(rest) is { Success: true } lstm)
        {
            string n = lstm.Groups[2].Value;
            collector.Add(HfWeightNormFolding.Sum(tensor.ToWeightInfo(), tensors[$"audio_encoder.{lstm.Groups[1].Value}.bias_hh_l{n}"].ToWeightInfo(), $"codec.{lstm.Groups[1].Value}.bias_l{n}"));
        }
        else if (rest.Contains(".lstm.bias_hh_l", StringComparison.Ordinal))
        {
            return;
        }
        else if (rest.Contains(".lstm.weight_", StringComparison.Ordinal))
        {
            collector.Add(HfTransforms.Plain(tensor.ToWeightInfo(), "codec." + rest));
        }
        else if (Codebook().Match(rest) is { Success: true } book)
        {
            if (int.Parse(book.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture) < codebooks)
            {
                collector.Add(HfTransforms.Plain(tensor.ToWeightInfo(), "codec." + rest));
            }
        }
        else
        {
            collector.AddUnmapped(name);
        }
    }

    private static string? Map(string name)
    {
        switch (name)
        {
            case "enc_to_dec_proj.weight" or "enc_to_dec_proj.bias":
                return "text.projection." + name["enc_to_dec_proj.".Length..];
            case "text_encoder.shared.weight":
                return "text.token_embedding.weight";
            case "text_encoder.encoder.embed_tokens.weight":
                return null;
            case "text_encoder.encoder.final_layer_norm.weight":
                return "text.output_norm.weight";
            case "decoder.model.decoder.embed_positions.weights":
                return "decoder.position_table.weight";
            case "decoder.model.decoder.layer_norm.weight" or "decoder.model.decoder.layer_norm.bias":
                return "decoder.output_norm." + name[(name.LastIndexOf('.') + 1)..];
        }

        if (HeadName().Match(name) is { Success: true } head)
        {
            return $"decoder.lm_head.{head.Groups[1].Value}.weight";
        }

        if (EmbeddingName().Match(name) is { Success: true } embedding)
        {
            return $"decoder.embedding.{embedding.Groups[1].Value}.weight";
        }

        if (DecoderLayer().Match(name) is { Success: true } layer)
        {
            string module = layer.Groups[2].Value switch
            {
                "self_attn.q_proj" => "self_attention.q",
                "self_attn.k_proj" => "self_attention.k",
                "self_attn.v_proj" => "self_attention.v",
                "self_attn.out_proj" => "self_attention.o",
                "encoder_attn.q_proj" => "cross_attention.q",
                "encoder_attn.k_proj" => "cross_attention.k",
                "encoder_attn.v_proj" => "cross_attention.v",
                "encoder_attn.out_proj" => "cross_attention.o",
                "self_attn_layer_norm" => "self_norm",
                "encoder_attn_layer_norm" => "cross_norm",
                "final_layer_norm" => "mlp_norm",
                "fc1" => "mlp.up",
                "fc2" => "mlp.down",
                _ => "",
            };
            return module.Length == 0 ? null : $"decoder.layers.{layer.Groups[1].Value}.{module}.{layer.Groups[3].Value}";
        }

        if (TextBlock().Match(name) is { Success: true } block)
        {
            string module = block.Groups[2].Value switch
            {
                "0.SelfAttention.q" => "attention.q.weight",
                "0.SelfAttention.k" => "attention.k.weight",
                "0.SelfAttention.v" => "attention.v.weight",
                "0.SelfAttention.o" => "attention.o.weight",
                "0.layer_norm" => "attention_norm.weight",
                "1.layer_norm" => "mlp_norm.weight",
                "1.DenseReluDense.wi" => "mlp.up.weight",
                "1.DenseReluDense.wo" => "mlp.down.weight",
                "0.SelfAttention.relative_attention_bias" => "",
                _ => null!,
            };
            if (module is null)
            {
                return null;
            }

            return module.Length == 0 ? "text.relative_bias.weight" : $"text.layers.{block.Groups[1].Value}.{module}";
        }

        return null;
    }

    [GeneratedRegex(@"^(.+)\.conv\.weight_(g|v)$")]
    private static partial Regex WeightNorm();

    [GeneratedRegex(@"^(.+\.lstm)\.bias_ih_l(\d+)$")]
    private static partial Regex LstmInput();

    [GeneratedRegex(@"^quantizer\.layers\.(\d+)\.codebook\.embed$")]
    private static partial Regex Codebook();

    [GeneratedRegex(@"^decoder\.lm_heads\.(\d+)\.weight$")]
    private static partial Regex HeadName();

    [GeneratedRegex(@"^decoder\.model\.decoder\.embed_tokens\.(\d+)\.weight$")]
    private static partial Regex EmbeddingName();

    [GeneratedRegex(@"^decoder\.model\.decoder\.layers\.(\d+)\.(.+)\.(weight|bias)$")]
    private static partial Regex DecoderLayer();

    [GeneratedRegex(@"^text_encoder\.encoder\.block\.(\d+)\.layer\.(\d\.[A-Za-z_.]+?)\.weight$")]
    private static partial Regex TextBlock();

    private static JsonElement Sub(HfConfig c, string key) => c.Root.TryGetProperty(key, out var value) && value.ValueKind == JsonValueKind.Object ? value : default;

    private static double Num(JsonElement e, string key, double fallback) => e.ValueKind == JsonValueKind.Object && e.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetDouble() : fallback;

    private static bool Bool(JsonElement e, string key, bool fallback) => e.ValueKind == JsonValueKind.Object && e.TryGetProperty(key, out var v) && v.ValueKind is JsonValueKind.True or JsonValueKind.False ? v.GetBoolean() : fallback;

    private static string Str(JsonElement e, string key, string fallback) => e.ValueKind == JsonValueKind.Object && e.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString()! : fallback;
}
