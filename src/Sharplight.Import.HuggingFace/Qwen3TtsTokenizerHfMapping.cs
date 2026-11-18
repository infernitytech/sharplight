using System.Text.Json;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Qwen3-TTS-Tokenizer-12Hz (<c>qwen3_tts_tokenizer_12hz</c>: the speech codec of Qwen3-TTS; a Mimi encoder that emits the first 16 of its 32 codebooks, and a decoder of a split
/// residual quantizer, a sliding-window transformer, ConvNeXt upsampling and a snake-activated SEANet-style vocoder; 24 kHz audio at 12.5 frames per second) → canonical
/// <c>qwen3_tts_codec</c> (documented on <c>Qwen3TtsCodecAdapter</c>). The encoder tensors map as Mimi's (<c>encoder.</c> prefix removed), the decoder's as they are without the
/// convolution modules' <c>.conv.</c> infix; the decoder's codebooks are computed from the EMA statistics.
/// </summary>
internal sealed class Qwen3TtsTokenizerHfMapping : IHfArchitectureMapping
{
    /// <inheritdoc/>
    public string Architecture => "qwen3_tts_codec";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["qwen3_tts_tokenizer_12hz"];

    /// <inheritdoc/>
    public bool NeedsTokenizer => false;

    /// <inheritdoc/>
    public IEnumerable<string> Validate(HfConfig c)
    {
        var encoder = c.Root.GetProperty("encoder_config");
        if (encoder.TryGetProperty("use_causal_conv", out var causal) && causal.ValueKind == JsonValueKind.False)
        {
            yield return "qwen3_tts_tokenizer_12hz: only a causal Mimi encoder is supported.";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HuggingFaceImportOptions options)
    {
        var e = c.Root.GetProperty("encoder_config");
        var d = c.Root.GetProperty("decoder_config");
        string a = Architecture + ".";
        int valid = c.Int("encoder_valid_num_quantizers");
        return new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "sample_rate"] = e.GetProperty("sampling_rate").GetInt32(),
            [a + "frame_rate"] = (float)e.GetProperty("_frame_rate").GetDouble(),
            [a + "hidden_size"] = e.GetProperty("hidden_size").GetInt32(),
            [a + "num_filters"] = e.GetProperty("num_filters").GetInt32(),
            [a + "kernel_size"] = e.GetProperty("kernel_size").GetInt32(),
            [a + "last_kernel_size"] = e.GetProperty("last_kernel_size").GetInt32(),
            [a + "residual_kernel_size"] = e.GetProperty("residual_kernel_size").GetInt32(),
            [a + "dilation_growth_rate"] = e.GetProperty("dilation_growth_rate").GetInt32(),
            [a + "residual_layers"] = e.GetProperty("num_residual_layers").GetInt32(),
            [a + "compress"] = e.GetProperty("compress").GetInt32(),
            [a + "upsampling_ratios"] = e.GetProperty("upsampling_ratios").EnumerateArray().Select(v => v.GetInt64()).ToArray(),
            [a + "codebook_size"] = e.GetProperty("codebook_size").GetInt32(),
            [a + "codebook_dim"] = e.GetProperty("vector_quantization_hidden_dimension").GetInt32(),
            [a + "semantic_codebooks"] = e.GetProperty("num_semantic_quantizers").GetInt32(),
            [a + "codebook_count"] = valid,
            [a + "default_codebooks"] = valid,
            [a + "transformer.layers"] = e.GetProperty("num_hidden_layers").GetInt32(),
            [a + "transformer.heads"] = e.GetProperty("num_attention_heads").GetInt32(),
            [a + "transformer.head_dim"] = e.GetProperty("head_dim").GetInt32(),
            [a + "transformer.norm_eps"] = (float)e.GetProperty("norm_eps").GetDouble(),
            [a + "transformer.sliding_window"] = e.GetProperty("sliding_window").GetInt32(),
            [a + "transformer.activation"] = e.GetProperty("hidden_act").GetString()!,
            [a + "transformer.rope_base"] = (float)e.GetProperty("rope_theta").GetDouble(),
            [a + "decoder.hidden_size"] = d.GetProperty("hidden_size").GetInt32(),
            [a + "decoder.latent_dim"] = d.GetProperty("latent_dim").GetInt32(),
            [a + "decoder.codebook_dim"] = d.GetProperty("codebook_dim").GetInt32(),
            [a + "decoder.codebook_size"] = d.GetProperty("codebook_size").GetInt32(),
            [a + "decoder.dim"] = d.GetProperty("decoder_dim").GetInt32(),
            [a + "decoder.layers"] = d.GetProperty("num_hidden_layers").GetInt32(),
            [a + "decoder.heads"] = d.GetProperty("num_attention_heads").GetInt32(),
            [a + "decoder.head_dim"] = d.GetProperty("head_dim").GetInt32(),
            [a + "decoder.norm_eps"] = (float)d.GetProperty("rms_norm_eps").GetDouble(),
            [a + "decoder.sliding_window"] = d.GetProperty("sliding_window").GetInt32(),
            [a + "decoder.rope_base"] = (float)d.GetProperty("rope_theta").GetDouble(),
            [a + "decoder.quantizers"] = d.GetProperty("num_quantizers").GetInt32(),
            [a + "decoder.semantic_quantizers"] = d.GetProperty("num_semantic_quantizers").GetInt32(),
            [a + "decoder.upsample_rates"] = d.GetProperty("upsample_rates").EnumerateArray().Select(v => v.GetInt64()).ToArray(),
            [a + "decoder.upsampling_ratios"] = d.GetProperty("upsampling_ratios").EnumerateArray().Select(v => v.GetInt64()).ToArray(),
        };
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
    {
        var collector = new HfWeightCollector();
        const string EncoderPrefix = "encoder.";
        var encoder = tensors.Where(t => t.Key.StartsWith(EncoderPrefix, StringComparison.Ordinal))
            .ToDictionary(t => t.Key[EncoderPrefix.Length..], t => t.Value, StringComparer.Ordinal);
        MimiHfMapping.MapInto(collector, encoder);
        foreach (var (name, tensor) in tensors)
        {
            if (name.StartsWith(EncoderPrefix, StringComparison.Ordinal))
            {
                continue;
            }

            string[] parts = name.Split('.');
            if (parts.Length > 3 && parts[0] == "decoder" && parts[1] == "quantizer" && parts[3] == "vq")
            {
                // decoder.quantizer.rvq_{first,rest}.vq.layers.{q}._codebook.{embedding_sum,cluster_usage}
                if (parts[^1] == "embedding_sum")
                {
                    string stem = name[..^"embedding_sum".Length];
                    var usage = tensors[stem + "cluster_usage"].ToWeightInfo();
                    var sum = tensor.ToWeightInfo();
                    string group = parts[2] == "rvq_first" ? "first" : "rest";
                    collector.Add(WeightInfo.Derived($"decoder.quantizer.{group}.layers.{parts[5]}.embed", DataType.F32, sum.Shape, () => Codebook(sum, usage)));
                }

                continue;
            }

            string target = name.Replace("rvq_first", "first", StringComparison.Ordinal).Replace("rvq_rest", "rest", StringComparison.Ordinal).Replace(".conv.", ".", StringComparison.Ordinal);
            collector.Add(HfTransforms.Plain(tensor.ToWeightInfo(), target));
        }

        collector.ThrowIfUnmapped(c.ModelType);
        return collector.Weights;
    }

    private static byte[] Codebook(WeightInfo sum, WeightInfo usage)
    {
        var values = HfTransforms.DecodeF32(sum);
        var counts = HfTransforms.DecodeF32(usage);
        long width = sum.Shape[1].Fixed;
        for (int row = 0; row < counts.Length; row++)
        {
            float divisor = MathF.Max(counts[row], 1e-5f);
            for (long i = 0; i < width; i++)
            {
                values[(row * width) + i] /= divisor;
            }
        }

        return HfTransforms.EncodeF32(values);
    }
}
