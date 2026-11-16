using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.RegularExpressions;
using Sharplight.Architectures;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Hugging Face CLAP (<c>clap</c>: ClapModel, e.g. laion/clap-htsat-unfused) → canonical <c>clap</c> (documented on <c>ClapAdapter</c>). The mel batch normalization is
/// folded into a scale and a shift, the 4 × 4 patch convolution becomes a matrix, the text tower keeps RoBERTa's post-norm layers, and the log-mel filter bank
/// (Slaney scale and normalization, the feature extractor's frequency range) is computed from the preprocessor configuration. The fusion variants (long audio with
/// local mel crops) are rejected.
/// </summary>
internal sealed partial class ClapHfMapping : IHfArchitectureMapping
{
    private const string AudioPrefix = "audio_model.audio_encoder.";

    /// <inheritdoc/>
    public string Architecture => "clap";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["clap"];

    /// <inheritdoc/>
    public IEnumerable<string> Validate(HfConfig c)
    {
        var audio = Part(c, "audio_config");
        if (audio.Bool("enable_fusion", false))
        {
            yield return "clap: the fusion variants (enable_fusion) are not supported.";
        }

        if (audio.String("hidden_act", "gelu") != "gelu" || Part(c, "text_config").String("hidden_act", "gelu") != "gelu")
        {
            yield return "clap: only gelu activations are supported.";
        }

        if (c.GetString("projection_hidden_act") is { } act && act != "relu")
        {
            yield return "clap: only relu projection activations are supported.";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HuggingFaceImportOptions options)
    {
        var audio = Part(c, "audio_config");
        var text = Part(c, "text_config");
        var feature = FeatureConfig(c.Directory);
        string a = Architecture + ".";
        int sampleRate = (int)feature.Double("sampling_rate", 48000);
        int hop = (int)feature.Double("hop_length", 480);
        return new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "sample_rate"] = sampleRate,
            [a + "clip_samples"] = (int)feature.Double("nb_max_samples", sampleRate * feature.Double("max_length_s", 10)),
            [a + "fft_size"] = (int)feature.Double("fft_window_size", 1024),
            [a + "hop_length"] = hop,
            [a + "mel_bins"] = audio.Int("num_mel_bins", 64),
            [a + "embedding_length"] = c.GetInt("projection_dim") ?? 512,
            [a + "audio.spec_size"] = audio.Int("spec_size", 256),
            [a + "audio.patch_size"] = audio.Int("patch_size", 4),
            [a + "audio.window_size"] = audio.Int("window_size", 8),
            [a + "audio.embed_dim"] = audio.Int("patch_embeds_hidden_size", 96),
            [a + "audio.depths"] = audio.Ints("depths", [2, 2, 6, 2]),
            [a + "audio.head_counts"] = audio.Ints("num_attention_heads", [4, 8, 16, 32]),
            [a + "audio.mlp_ratio"] = (int)audio.Double("mlp_ratio", 4.0),
            [a + "audio.norm_eps"] = (float)audio.Double("layer_norm_eps", 1e-5),
            [a + "text.layer_count"] = text.Int("num_hidden_layers", 12),
            [a + "text.hidden_size"] = text.Int("hidden_size", 768),
            [a + "text.attention.head_count"] = text.Int("num_attention_heads", 12),
            [a + "text.norm_eps"] = (float)text.Double("layer_norm_eps", 1e-12),
            [a + "text.context_length"] = text.Int("max_position_embeddings", 514) - text.Int("pad_token_id", 1) - 1,
            [a + "text.position_offset"] = text.Int("pad_token_id", 1) + 1,
        };
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
    {
        var audio = Part(c, "audio_config");
        int embed = audio.Int("patch_embeds_hidden_size", 96);
        float eps = 1e-5f;
        var collector = new HfWeightCollector();
        foreach (var (name, tensor) in tensors)
        {
            var info = tensor.ToWeightInfo();
            switch (name)
            {
                case "logit_scale_a":
                    collector.Add(HfTransforms.ToF32(info, "logit_scale_audio", new Shape(1)));
                    continue;
                case "logit_scale_t":
                    collector.Add(HfTransforms.ToF32(info, "logit_scale_text", new Shape(1)));
                    continue;
                case AudioPrefix + "batch_norm.weight":
                    foreach (var folded in FoldBatchNorm(tensors, eps))
                    {
                        collector.Add(folded);
                    }

                    continue;
                case AudioPrefix + "batch_norm.bias" or AudioPrefix + "batch_norm.running_mean" or AudioPrefix + "batch_norm.running_var" or AudioPrefix + "batch_norm.num_batches_tracked":
                    continue;
                case AudioPrefix + "patch_embed.proj.weight":
                    collector.Add(info.As("audio.patch_embed.weight", new Shape(embed, info.Shape.Elements() / embed)));
                    continue;
                case AudioPrefix + "patch_embed.proj.bias":
                    collector.Add(HfTransforms.Plain(info, "audio.patch_embed.bias"));
                    continue;
                case "text_model.embeddings.word_embeddings.weight":
                    collector.Add(info.As("text.token_embedding.weight"));
                    continue;
                case "text_model.embeddings.position_embeddings.weight":
                    collector.Add(HfTransforms.ToF32(info, "text.position_embedding.weight"));
                    continue;
                case "text_model.embeddings.token_type_embeddings.weight":
                    collector.Add(HfTransforms.ToF32(info, "text.token_type_embedding.weight"));
                    continue;
                case var _ when name.EndsWith("embeddings.position_ids", StringComparison.Ordinal) || name.EndsWith("embeddings.token_type_ids", StringComparison.Ordinal)
                    || name.EndsWith("relative_position_index", StringComparison.Ordinal):
                    continue;
            }

            string? mapped = MapName(name);
            if (mapped is null)
            {
                collector.AddUnmapped(name);
                continue;
            }

            collector.Add(mapped.EndsWith("relative_bias", StringComparison.Ordinal) ? HfTransforms.ToF32(info, mapped) : info.Shape.Rank <= 1 ? HfTransforms.Plain(info, mapped) : info.As(mapped));
        }

        collector.ThrowIfUnmapped(c.ModelType);
        return collector.Weights;
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, IWeightSource> MapAttachments(HfConfig config, string directory)
    {
        var feature = FeatureConfig(directory);
        int bins = (int)feature.Double("feature_size", 64);
        int fft = (int)feature.Double("fft_window_size", 1024);
        var filters = HfMelFilterBank.Create(bins, fft, (int)feature.Double("sampling_rate", 48000), feature.Double("frequency_min", 50), feature.Double("frequency_max", 14000));
        return new Dictionary<string, IWeightSource> { [ClapAdapter.MelFiltersAttachment] = new MemoryWeightSource(MemoryMarshal.AsBytes(filters.AsSpan()).ToArray()) };
    }

    private static string? MapName(string name)
    {
        if (name.StartsWith(AudioPrefix, StringComparison.Ordinal))
        {
            string rest = name[AudioPrefix.Length..];
            if (rest.StartsWith("patch_embed.norm.", StringComparison.Ordinal))
            {
                return "audio.patch_norm." + rest["patch_embed.norm.".Length..];
            }

            if (rest.StartsWith("norm.", StringComparison.Ordinal))
            {
                return "audio." + rest;
            }

            var block = Block().Match(rest);
            if (block.Success)
            {
                string module = block.Groups[3].Value switch
                {
                    "layernorm_before" => "norm1",
                    "layernorm_after" => "norm2",
                    "attention.self.query" => "q",
                    "attention.self.key" => "k",
                    "attention.self.value" => "v",
                    "attention.output.dense" => "o",
                    "intermediate.dense" => "mlp.up",
                    "output.dense" => "mlp.down",
                    "attention.self" when block.Groups[4].Value == "relative_position_bias_table" => "relative_bias",
                    _ => "",
                };
                if (module.Length == 0)
                {
                    return null;
                }

                return module == "relative_bias" ? $"audio.layers.{block.Groups[1].Value}.blocks.{block.Groups[2].Value}.relative_bias"
                    : $"audio.layers.{block.Groups[1].Value}.blocks.{block.Groups[2].Value}.{module}.{block.Groups[4].Value}";
            }

            var down = Downsample().Match(rest);
            if (down.Success)
            {
                return down.Groups[2].Value == "norm" ? $"audio.layers.{down.Groups[1].Value}.merge_norm.{down.Groups[3].Value}" : $"audio.layers.{down.Groups[1].Value}.merge.{down.Groups[3].Value}";
            }

            return null;
        }

        if (Projection().Match(name) is { Success: true } projection)
        {
            string tower = projection.Groups[1].Value == "audio" ? "audio.projection" : "text.projection";
            return $"{tower}.{projection.Groups[2].Value}.{projection.Groups[3].Value}";
        }

        if (name.StartsWith("text_model.embeddings.LayerNorm.", StringComparison.Ordinal))
        {
            return "text.embedding_norm." + name["text_model.embeddings.LayerNorm.".Length..];
        }

        if (name.StartsWith("text_model.pooler.dense.", StringComparison.Ordinal))
        {
            return "text.pooler." + name["text_model.pooler.dense.".Length..];
        }

        var layer = TextLayer().Match(name);
        if (layer.Success)
        {
            string module = layer.Groups[2].Value switch
            {
                "attention.self.query" => "attention.q",
                "attention.self.key" => "attention.k",
                "attention.self.value" => "attention.v",
                "attention.output.dense" => "attention.o",
                "attention.output.LayerNorm" => "attention_norm",
                "intermediate.dense" => "mlp.up",
                "output.dense" => "mlp.down",
                "output.LayerNorm" => "mlp_norm",
                _ => "",
            };
            return module.Length == 0 ? null : $"text.layers.{layer.Groups[1].Value}.{module}.{layer.Groups[3].Value}";
        }

        return null;
    }

    // y = (x - mean) / sqrt(var + eps) · gamma + beta as x · scale + shift.
    private static IEnumerable<WeightInfo> FoldBatchNorm(IReadOnlyDictionary<string, SafetensorsTensor> tensors, float eps)
    {
        var gamma = tensors[AudioPrefix + "batch_norm.weight"].ToWeightInfo();
        var beta = tensors[AudioPrefix + "batch_norm.bias"].ToWeightInfo();
        var mean = tensors[AudioPrefix + "batch_norm.running_mean"].ToWeightInfo();
        var variance = tensors[AudioPrefix + "batch_norm.running_var"].ToWeightInfo();
        var shape = gamma.Shape;
        float[]? scale = null;
        float[] Scale() => scale ??= Compute();
        float[] Compute()
        {
            var g = HfTransforms.DecodeF32(gamma);
            var v = HfTransforms.DecodeF32(variance);
            var result = new float[g.Length];
            for (int i = 0; i < g.Length; i++)
            {
                result[i] = (float)(g[i] / Math.Sqrt(v[i] + (double)eps));
            }

            return result;
        }

        yield return WeightInfo.Derived("audio.batch_norm.scale", DataType.F32, shape, () => HfTransforms.EncodeF32(Scale()));
        yield return WeightInfo.Derived("audio.batch_norm.shift", DataType.F32, shape, () =>
        {
            var s = Scale();
            var b = HfTransforms.DecodeF32(beta);
            var m = HfTransforms.DecodeF32(mean);
            var result = new float[s.Length];
            for (int i = 0; i < s.Length; i++)
            {
                result[i] = b[i] - (m[i] * s[i]);
            }

            return HfTransforms.EncodeF32(result);
        });
    }

    [GeneratedRegex(@"^layers\.(\d+)\.blocks\.(\d+)\.(.+)\.(weight|bias|relative_position_bias_table)$")]
    private static partial Regex Block();

    [GeneratedRegex(@"^layers\.(\d+)\.downsample\.(norm|reduction)\.(weight|bias)$")]
    private static partial Regex Downsample();

    [GeneratedRegex(@"^(audio|text)_projection\.linear(\d)\.(weight|bias)$")]
    private static partial Regex Projection();

    [GeneratedRegex(@"^text_model\.encoder\.layer\.(\d+)\.(.+)\.(weight|bias)$")]
    private static partial Regex TextLayer();

    private static Section Part(HfConfig c, string key)
        => new(c.Root.TryGetProperty(key, out var value) && value.ValueKind == JsonValueKind.Object ? value : default);

    private static Section FeatureConfig(string directory)
    {
        string path = Path.Combine(directory, "preprocessor_config.json");
        if (!File.Exists(path))
        {
            return new Section(default);
        }

        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        return new Section(doc.RootElement.Clone());
    }

    private readonly record struct Section(JsonElement Value)
    {
        public int Int(string key, int fallback) => Has(key) ? (int)Value.GetProperty(key).GetDouble() : fallback;

        public double Double(string key, double fallback) => Has(key) ? Value.GetProperty(key).GetDouble() : fallback;

        public bool Bool(string key, bool fallback) => Has(key) ? Value.GetProperty(key).ValueKind == JsonValueKind.True : fallback;

        public string String(string key, string fallback) => Has(key) && Value.GetProperty(key).ValueKind == JsonValueKind.String ? Value.GetProperty(key).GetString()! : fallback;

        public int[] Ints(string key, int[] fallback) => Has(key) ? [.. Value.GetProperty(key).EnumerateArray().Select(v => v.GetInt32())] : fallback;

        private bool Has(string key) => Value.ValueKind == JsonValueKind.Object && Value.TryGetProperty(key, out var v) && v.ValueKind != JsonValueKind.Null;
    }
}
