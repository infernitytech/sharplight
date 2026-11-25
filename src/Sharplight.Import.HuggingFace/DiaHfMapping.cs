using System.Text.Json;
using Sharplight.Models;
using Sharplight.Tensors;
using Sharplight.Vision;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Dia (<c>dia</c>: Nari Labs' dialogue text-to-speech model, an encoder-decoder transformer over the nine DAC codebooks) → canonical <c>dia</c> (documented on <c>DiaAdapter</c>).
/// Two layouts of the same network are read: the transformers port (<c>model_type</c> <c>dia</c>, as in Dia-1.6B-0626: fused gate/up projections, one embedding table, the
/// projections as [out, in]) and the original checkpoint of the <c>dia</c> package (Dia-1.6B: a <c>config.json</c> with <c>data</c> and <c>model</c> sections and no model type; projections
/// as [in, heads, head dim], a fused <c>wi_fused</c> [in, 2, hidden], one embedding table per codebook, heads as [hidden, channels, vocabulary]), whose tensors are transposed on first read.
/// The text needs no tokenizer (bytes). The 44.1 kHz DAC codec is the model's vocoder companion, a <c>dac/</c> (or <c>audio_tokenizer/</c>) folder of Hugging Face DAC files
/// (<see cref="DacHfMapping"/>).
/// </summary>
internal sealed class DiaHfMapping : IHfArchitectureMapping
{
    /// <summary>The model type reported for both layouts.</summary>
    public const string ModelType = "dia";

    /// <inheritdoc/>
    public string Architecture => "dia";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = [ModelType];

    /// <inheritdoc/>
    public bool NeedsTokenizer => false;

    /// <summary>Whether <paramref name="config"/> is the original checkpoint's (a <c>data</c> and a <c>model</c> section and no model type).</summary>
    /// <param name="config">The checkpoint's configuration.</param>
    public static bool IsOriginal(HfConfig config)
        => config.Root.ValueKind == JsonValueKind.Object && !config.Root.TryGetProperty("model_type", out _) && config.Root.TryGetProperty("data", out var data)
           && data.ValueKind == JsonValueKind.Object && data.TryGetProperty("delay_pattern", out _) && config.Root.TryGetProperty("model", out var model)
           && model.ValueKind == JsonValueKind.Object && model.TryGetProperty("decoder", out _);

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HuggingFaceImportOptions options)
    {
        var md = new Dictionary<string, object>(StringComparer.Ordinal);
        string a = Architecture + ".";
        if (IsOriginal(c))
        {
            var data = c.Root.GetProperty("data");
            var model = c.Root.GetProperty("model");
            var decoder = model.GetProperty("decoder");
            var encoder = model.GetProperty("encoder");
            md[a + "layer_count"] = decoder.GetProperty("n_layer").GetInt32();
            md[a + "hidden_size"] = decoder.GetProperty("n_embd").GetInt32();
            md[a + "attention.head_count"] = decoder.GetProperty("gqa_query_heads").GetInt32();
            md[a + "attention.head_count_kv"] = decoder.GetProperty("kv_heads").GetInt32();
            md[a + "head_dim"] = decoder.GetProperty("gqa_head_dim").GetInt32();
            md[a + "cross_attention.head_count"] = decoder.GetProperty("cross_query_heads").GetInt32();
            md[a + "cross_attention.head_dim"] = decoder.GetProperty("cross_head_dim").GetInt32();
            md[a + "intermediate_size"] = decoder.GetProperty("n_hidden").GetInt32();
            md[a + "channels"] = data.GetProperty("channels").GetInt32();
            md[a + "vocabulary"] = model.GetProperty("tgt_vocab_size").GetInt32();
            md[a + "context_length"] = data.GetProperty("audio_length").GetInt32();
            md[a + "norm_eps"] = (float)model.GetProperty("normalization_layer_epsilon").GetDouble();
            md[a + "rope.base"] = (float)model.GetProperty("rope_max_timescale").GetDouble();
            md[a + "delay_pattern"] = Longs(data.GetProperty("delay_pattern"));
            md[a + "bos_token"] = data.GetProperty("audio_bos_value").GetInt32();
            md[a + "eos_token"] = data.GetProperty("audio_eos_value").GetInt32();
            md[a + "pad_token"] = data.GetProperty("audio_pad_value").GetInt32();
            md[a + "encoder.layer_count"] = encoder.GetProperty("n_layer").GetInt32();
            md[a + "encoder.hidden_size"] = encoder.GetProperty("n_embd").GetInt32();
            md[a + "encoder.attention.head_count"] = encoder.GetProperty("n_head").GetInt32();
            md[a + "encoder.head_dim"] = encoder.GetProperty("head_dim").GetInt32();
            md[a + "encoder.intermediate_size"] = encoder.GetProperty("n_hidden").GetInt32();
            md[a + "encoder.vocabulary"] = model.GetProperty("src_vocab_size").GetInt32();
            md[a + "encoder.context_length"] = data.GetProperty("text_length").GetInt32();

            // The dia package's generate() defaults.
            md[a + "sampling.guidance_scale"] = 3f;
            md[a + "sampling.temperature"] = 1.3f;
            md[a + "sampling.top_k"] = 35f;
            md[a + "sampling.top_p"] = 0.95f;
            return md;
        }

        var d = c.Root.GetProperty("decoder_config");
        var e = c.Root.GetProperty("encoder_config");
        md[a + "layer_count"] = d.GetProperty("num_hidden_layers").GetInt32();
        md[a + "hidden_size"] = d.GetProperty("hidden_size").GetInt32();
        md[a + "attention.head_count"] = d.GetProperty("num_attention_heads").GetInt32();
        md[a + "attention.head_count_kv"] = d.GetProperty("num_key_value_heads").GetInt32();
        md[a + "head_dim"] = d.GetProperty("head_dim").GetInt32();
        md[a + "cross_attention.head_count"] = d.GetProperty("cross_num_attention_heads").GetInt32();
        md[a + "cross_attention.head_dim"] = d.GetProperty("cross_head_dim").GetInt32();
        md[a + "intermediate_size"] = d.GetProperty("intermediate_size").GetInt32();
        md[a + "channels"] = d.GetProperty("num_channels").GetInt32();
        md[a + "vocabulary"] = d.GetProperty("vocab_size").GetInt32();
        md[a + "context_length"] = d.GetProperty("max_position_embeddings").GetInt32();
        md[a + "norm_eps"] = (float)d.GetProperty("norm_eps").GetDouble();
        md[a + "rope.base"] = (float)d.GetProperty("rope_theta").GetDouble();
        md[a + "delay_pattern"] = Longs(c.Root.GetProperty("delay_pattern"));
        md[a + "bos_token"] = c.Root.GetProperty("bos_token_id").GetInt32();
        md[a + "eos_token"] = c.Root.GetProperty("eos_token_id").GetInt32();
        md[a + "pad_token"] = c.Root.GetProperty("pad_token_id").GetInt32();
        md[a + "encoder.layer_count"] = e.GetProperty("num_hidden_layers").GetInt32();
        md[a + "encoder.hidden_size"] = e.GetProperty("hidden_size").GetInt32();
        md[a + "encoder.attention.head_count"] = e.GetProperty("num_attention_heads").GetInt32();
        md[a + "encoder.head_dim"] = e.GetProperty("head_dim").GetInt32();
        md[a + "encoder.intermediate_size"] = e.GetProperty("intermediate_size").GetInt32();
        md[a + "encoder.vocabulary"] = e.GetProperty("vocab_size").GetInt32();
        md[a + "encoder.context_length"] = e.GetProperty("max_position_embeddings").GetInt32();
        string generation = Path.Combine(c.Directory, "generation_config.json");
        if (File.Exists(generation))
        {
            using var g = JsonDocument.Parse(File.ReadAllText(generation));
            foreach (var (key, name) in new[] { ("guidance_scale", "guidance_scale"), ("temperature", "temperature"), ("top_k", "top_k"), ("top_p", "top_p") })
            {
                if (g.RootElement.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.Number)
                {
                    md[a + "sampling." + name] = (float)v.GetDouble();
                }
            }
        }

        return md;
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
        => IsOriginal(c) ? MapOriginal(c, tensors) : MapTransformers(c, tensors);

    /// <summary>The DAC codec folder (<c>dac/</c> or <c>audio_tokenizer/</c>) as the vocoder companion.</summary>
    /// <param name="directory">The model folder.</param>
    public IReadOnlyDictionary<string, string> FindCompanions(string directory)
    {
        foreach (string name in new[] { "dac", "audio_tokenizer" })
        {
            string codec = Path.Combine(directory, name);
            if (File.Exists(Path.Combine(codec, "config.json")))
            {
                return new Dictionary<string, string> { [AuxiliaryRoles.Vocoder] = codec };
            }
        }

        return new Dictionary<string, string>();
    }

    private static IReadOnlyDictionary<string, WeightInfo> MapTransformers(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors)
    {
        var collector = new HfWeightCollector();
        int encoderIntermediate = c.Root.GetProperty("encoder_config").GetProperty("intermediate_size").GetInt32();
        int decoderIntermediate = c.Root.GetProperty("decoder_config").GetProperty("intermediate_size").GetInt32();
        foreach (var (name, tensor) in tensors)
        {
            var w = tensor.ToWeightInfo();
            string? mapped = name switch
            {
                "model.encoder.embedding.weight" => "encoder.token_embedding.weight",
                "model.encoder.norm.weight" => "encoder.output_norm.weight",
                "model.decoder.embeddings.embed.weight" => "decoder.embedding.weight",
                "model.decoder.norm.weight" => "decoder.output_norm.weight",
                "logits_dense.weight" => "decoder.lm_head.weight",
                _ => null,
            };
            if (mapped is not null)
            {
                collector.Add(HfTransforms.Plain(w, mapped));
                continue;
            }

            bool encoder = name.StartsWith("model.encoder.layers.", StringComparison.Ordinal);
            bool decoder = name.StartsWith("model.decoder.layers.", StringComparison.Ordinal);
            if (!encoder && !decoder)
            {
                collector.AddUnmapped(name);
                continue;
            }

            string stack = encoder ? "encoder" : "decoder";
            string rest = name[(name.IndexOf('.', "model.encoder.layers.".Length) + 1)..];
            string layer = name["model.encoder.layers.".Length..name.IndexOf('.', "model.encoder.layers.".Length)];
            string prefix = $"{stack}.layers.{layer}.";
            switch (rest)
            {
                case "pre_sa_norm.weight":
                    collector.Add(HfTransforms.Plain(w, prefix + "attention_norm.weight"));
                    break;
                case "post_sa_norm.weight" or "pre_mlp_norm.weight":
                    collector.Add(HfTransforms.Plain(w, prefix + "mlp_norm.weight"));
                    break;
                case "pre_ca_norm.weight":
                    collector.Add(HfTransforms.Plain(w, prefix + "cross_attention_norm.weight"));
                    break;
                case "mlp.gate_up_proj.weight":
                    int intermediate = encoder ? encoderIntermediate : decoderIntermediate;
                    collector.Add(HfTransforms.Plain(w.RowSlice("", 0, intermediate), prefix + "mlp.gate.weight"));
                    collector.Add(HfTransforms.Plain(w.RowSlice("", intermediate, intermediate), prefix + "mlp.up.weight"));
                    break;
                case "mlp.down_proj.weight":
                    collector.Add(HfTransforms.Plain(w, prefix + "mlp.down.weight"));
                    break;
                default:
                    if (rest.StartsWith("self_attention.", StringComparison.Ordinal) && rest.EndsWith("_proj.weight", StringComparison.Ordinal))
                    {
                        collector.Add(HfTransforms.Plain(w, prefix + "attention." + rest["self_attention.".Length] + ".weight"));
                    }
                    else if (rest.StartsWith("cross_attention.", StringComparison.Ordinal) && rest.EndsWith("_proj.weight", StringComparison.Ordinal))
                    {
                        collector.Add(HfTransforms.Plain(w, prefix + "cross_attention." + rest["cross_attention.".Length] + ".weight"));
                    }
                    else
                    {
                        collector.AddUnmapped(name);
                    }

                    break;
            }
        }

        collector.ThrowIfUnmapped(c.ModelType);
        return collector.Weights;
    }

    private static IReadOnlyDictionary<string, WeightInfo> MapOriginal(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors)
    {
        var collector = new HfWeightCollector();
        var model = c.Root.GetProperty("model");
        int channels = c.Root.GetProperty("data").GetProperty("channels").GetInt32();
        var embeddings = new WeightInfo?[channels];
        foreach (var (name, tensor) in tensors)
        {
            var w = tensor.ToWeightInfo();
            string[] parts = name.Split('.');
            if (name == "encoder.embedding.weight")
            {
                collector.Add(HfTransforms.Plain(w, "encoder.token_embedding.weight"));
            }
            else if (name == "encoder.norm.weight")
            {
                collector.Add(HfTransforms.Plain(w, "encoder.output_norm.weight"));
            }
            else if (name == "decoder.norm.weight")
            {
                collector.Add(HfTransforms.Plain(w, "decoder.output_norm.weight"));
            }
            else if (name == "decoder.logits_dense.weight")
            {
                collector.Add(Transposed(w, "decoder.lm_head.weight", w.Shape[0].Fixed, w.Shape[1].Fixed * w.Shape[2].Fixed, 0, w.Shape[1].Fixed * w.Shape[2].Fixed));
            }
            else if (parts is ["decoder", "embeddings", var channel, "weight"])
            {
                embeddings[int.Parse(channel, System.Globalization.CultureInfo.InvariantCulture)] = w;
            }
            else if (parts is [var stack and ("encoder" or "decoder"), "layers", var layer, .. var rest])
            {
                MapOriginalLayer(collector, w, name, $"{stack}.layers.{layer}.", string.Join('.', rest), model, stack == "encoder");
            }
            else
            {
                collector.AddUnmapped(name);
            }
        }

        if (embeddings.Any(e => e is null))
        {
            throw new InvalidDataException("dia: a codebook embedding table is missing.");
        }

        collector.Add(WeightInfo.ConcatRows("decoder.embedding.weight", [.. embeddings.Select(e => e!)]));
        collector.ThrowIfUnmapped("dia");
        return collector.Weights;
    }

    private static void MapOriginalLayer(HfWeightCollector collector, WeightInfo w, string name, string prefix, string rest, JsonElement model, bool encoder)
    {
        int hidden = (encoder ? model.GetProperty("encoder").GetProperty("n_hidden") : model.GetProperty("decoder").GetProperty("n_hidden")).GetInt32();
        switch (rest)
        {
            case "pre_sa_norm.weight":
                collector.Add(HfTransforms.Plain(w, prefix + "attention_norm.weight"));
                break;
            case "post_sa_norm.weight" or "pre_mlp_norm.weight":
                collector.Add(HfTransforms.Plain(w, prefix + "mlp_norm.weight"));
                break;
            case "pre_ca_norm.weight":
                collector.Add(HfTransforms.Plain(w, prefix + "cross_attention_norm.weight"));
                break;
            case "mlp.wi_fused.weight":
            {
                long input = w.Shape[0].Fixed;
                collector.Add(Transposed(w, prefix + "mlp.gate.weight", input, 2 * hidden, 0, hidden));
                collector.Add(Transposed(w, prefix + "mlp.up.weight", input, 2 * hidden, hidden, hidden));
                break;
            }

            case "mlp.wo.weight":
                collector.Add(Transposed(w, prefix + "mlp.down.weight", w.Shape[0].Fixed, w.Shape[1].Fixed, 0, w.Shape[1].Fixed));
                break;
            default:
            {
                string[] parts = rest.Split('.');
                if (parts is [var attention and ("self_attention" or "cross_attention"), var projection, "weight"] && projection.EndsWith("_proj", StringComparison.Ordinal))
                {
                    string target = (attention == "self_attention" ? "attention." : "cross_attention.") + projection[0] + ".weight";
                    if (projection[0] == 'o')
                    {
                        // [heads, head dim, out] -> [out, heads * head dim]
                        collector.Add(Transposed(w, prefix + target, w.Shape[0].Fixed * w.Shape[1].Fixed, w.Shape[2].Fixed, 0, w.Shape[2].Fixed));
                    }
                    else
                    {
                        // [in, heads, head dim] -> [heads * head dim, in]
                        long outputs = w.Shape[1].Fixed * w.Shape[2].Fixed;
                        collector.Add(Transposed(w, prefix + target, w.Shape[0].Fixed, outputs, 0, outputs));
                    }
                }
                else
                {
                    collector.AddUnmapped(name);
                }

                break;
            }
        }
    }

    // Columns [firstColumn, firstColumn + columns) of the tensor viewed as [rows, totalColumns], transposed, as f32.
    private static WeightInfo Transposed(WeightInfo source, string name, long rows, long totalColumns, long firstColumn, long columns)
        => WeightInfo.Derived(name, DataType.F32, new Shape(columns, rows), () =>
        {
            var values = HfTransforms.DecodeF32(source);
            var result = new float[columns * rows];
            for (long r = 0; r < rows; r++)
            {
                for (long k = 0; k < columns; k++)
                {
                    result[(k * rows) + r] = values[(r * totalColumns) + firstColumn + k];
                }
            }

            return HfTransforms.EncodeF32(result);
        });

    private static long[] Longs(JsonElement array) => [.. array.EnumerateArray().Select(v => v.GetInt64())];
}
