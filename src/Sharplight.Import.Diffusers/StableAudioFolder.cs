using System.Text.Json;
using Sharplight.Import.HuggingFace;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Reads a diffusers Stable Audio pipeline folder (<c>StableAudioPipeline</c>: <c>transformer/</c> with a <c>StableAudioDiTModel</c>, <c>projection_model/</c> with the
/// timing conditioners, <c>vae/</c> with an <c>AutoencoderOobleck</c>, <c>text_encoder/</c> (T5), <c>tokenizer/</c>, <c>scheduler/</c> with a
/// <c>CosineDPMSolverMultistepScheduler</c>) into the canonical <c>stable_audio</c> source (vocabulary on <c>StableAudioAdapter</c>). Only the autoencoder's decoder is
/// imported.
/// </summary>
internal static class StableAudioFolder
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "stable_audio";

    private const string Prefix = Architecture + ".";

    /// <summary>Whether the folder holds a Stable Audio transformer, its timing conditioners and its autoencoder.</summary>
    /// <param name="path">A directory.</param>
    public static bool IsFolder(string path)
    {
        string config = Path.Combine(path, "transformer", "config.json");
        if (!File.Exists(config) || !Directory.Exists(Path.Combine(path, "projection_model")))
        {
            return false;
        }

        using var document = JsonDocument.Parse(File.ReadAllBytes(config));
        return document.RootElement.String("_class_name", "") == "StableAudioDiTModel";
    }

    /// <summary>Opens the folder as a canonical source.</summary>
    /// <param name="path">The pipeline folder.</param>
    public static ModelSource Open(string path)
    {
        var resources = new List<IDisposable>();
        try
        {
            var weights = new CanonicalWeights();
            var t5 = FluxSource.AddT5Folder(Path.Combine(path, "text_encoder"), Path.Combine(path, "tokenizer"), weights, resources, "text.");

            var projection = SafetensorsCheckpoint.Open(Path.Combine(path, "projection_model"));
            resources.Add(projection);
            foreach (var (name, tensor) in projection.Tensors)
            {
                AddTiming(name, tensor.ToWeightInfo(), weights);
            }

            var transformer = SafetensorsCheckpoint.Open(Path.Combine(path, "transformer"));
            resources.Add(transformer);
            foreach (var (name, tensor) in transformer.Tensors)
            {
                AddTransformer(name, tensor.ToWeightInfo(), weights);
            }

            var vae = SafetensorsCheckpoint.Open(Path.Combine(path, "vae"));
            resources.Add(vae);
            OobleckDecoderWeights.Map(vae.Tensors, "vae.", w => weights.Add(w.Name, w));

            var metadata = HyperParameters(path);
            metadata[CanonicalKeys.Name] = Path.GetFileName(Path.TrimEndingDirectorySeparator(Path.GetFullPath(path)));
            metadata[CanonicalKeys.Architecture] = Architecture;
            const string encoder = "encoder_decoder.";
            foreach (var (key, value) in t5)
            {
                if (key.StartsWith(encoder, StringComparison.Ordinal))
                {
                    metadata[Prefix + "text." + key[encoder.Length..]] = value;
                }
                else if (key.StartsWith("tokenizer.", StringComparison.Ordinal))
                {
                    metadata[key] = value;
                }
            }

            metadata[Prefix + "text.prompt_length"] = PromptLength(Path.Combine(path, "tokenizer"));
            return new ModelSource
            {
                Format = "diffusers",
                Architecture = Architecture,
                Metadata = metadata,
                Weights = weights.Weights,
                OwnedResources = resources,
                Path = path,
            };
        }
        catch
        {
            foreach (var resource in resources)
            {
                resource.Dispose();
            }

            throw;
        }
    }

    // start_number_conditioner.time_positional_embedding.{0.weights, 1.weight, 1.bias} → timing.start.{frequencies, projection.weight, projection.bias}; end → total.
    private static void AddTiming(string name, WeightInfo weight, CanonicalWeights weights)
    {
        string number = name.StartsWith("start_number_conditioner.", StringComparison.Ordinal) ? "start"
            : name.StartsWith("end_number_conditioner.", StringComparison.Ordinal) ? "total"
            : throw new NotSupportedException($"Stable Audio projection model: tensor '{name}' has no canonical mapping.");
        string module = name[(name.IndexOf('.') + 1)..] switch
        {
            "time_positional_embedding.0.weights" => "frequencies",
            "time_positional_embedding.1.weight" => "projection.weight",
            "time_positional_embedding.1.bias" => "projection.bias",
            _ => throw new NotSupportedException($"Stable Audio projection model: tensor '{name}' has no canonical mapping."),
        };
        weights.AddF32($"timing.{number}.{module}", weight);
    }

    private static void AddTransformer(string name, WeightInfo weight, CanonicalWeights weights)
    {
        const string block = "transformer_blocks.";
        if (name.StartsWith(block, StringComparison.Ordinal))
        {
            int dot = name.IndexOf('.', block.Length);
            string layer = name[block.Length..dot];
            string module = name[(dot + 1)..] switch
            {
                "norm1.weight" => "attention_norm.weight",
                "norm1.bias" => "attention_norm.bias",
                "attn1.to_q.weight" => "attention.q.weight",
                "attn1.to_k.weight" => "attention.k.weight",
                "attn1.to_v.weight" => "attention.v.weight",
                "attn1.to_out.0.weight" => "attention.o.weight",
                "norm2.weight" => "cross_attention_norm.weight",
                "norm2.bias" => "cross_attention_norm.bias",
                "attn2.to_q.weight" => "cross_attention.q.weight",
                "attn2.to_k.weight" => "cross_attention.k.weight",
                "attn2.to_v.weight" => "cross_attention.v.weight",
                "attn2.to_out.0.weight" => "cross_attention.o.weight",
                "norm3.weight" => "mlp_norm.weight",
                "norm3.bias" => "mlp_norm.bias",
                "ff.net.0.proj.weight" => "mlp.up.weight",
                "ff.net.0.proj.bias" => "mlp.up.bias",
                "ff.net.2.weight" => "mlp.down.weight",
                "ff.net.2.bias" => "mlp.down.bias",
                _ => throw new NotSupportedException($"Stable Audio transformer: tensor '{name}' has no canonical mapping."),
            };
            weights.Add($"dit.layers.{layer}.{module}", weight);
            return;
        }

        switch (name)
        {
            case "preprocess_conv.weight":
            case "postprocess_conv.weight":
                // The 1×1 convolutions mixing the latent channels, as [C, C] matrices.
                string mix = name.StartsWith("pre", StringComparison.Ordinal) ? "dit.input_mix.weight" : "dit.output_mix.weight";
                weights.AddF32(mix, weight.As(mix, new Shape(weight.Shape[0].Fixed, weight.Shape[1].Fixed)));
                return;
            case "time_proj.weight":
                weights.AddF32("dit.time_embedding.frequencies", weight);
                return;
        }

        string canonical = name switch
        {
            "timestep_proj.0.weight" => "dit.time_embedding.linear_1.weight",
            "timestep_proj.0.bias" => "dit.time_embedding.linear_1.bias",
            "timestep_proj.2.weight" => "dit.time_embedding.linear_2.weight",
            "timestep_proj.2.bias" => "dit.time_embedding.linear_2.bias",
            "global_proj.0.weight" => "dit.global_embedding.linear_1.weight",
            "global_proj.2.weight" => "dit.global_embedding.linear_2.weight",
            "cross_attention_proj.0.weight" => "dit.condition_embedding.linear_1.weight",
            "cross_attention_proj.2.weight" => "dit.condition_embedding.linear_2.weight",
            "proj_in.weight" => "dit.proj_in.weight",
            "proj_out.weight" => "dit.proj_out.weight",
            _ => throw new NotSupportedException($"Stable Audio transformer: tensor '{name}' has no canonical mapping."),
        };
        weights.Add(canonical, weight);
    }

    private static Dictionary<string, object> HyperParameters(string path)
    {
        var transformer = ReadJson(Path.Combine(path, "transformer", "config.json"));
        var projection = ReadJson(Path.Combine(path, "projection_model", "config.json"));
        var vae = ReadJson(Path.Combine(path, "vae", "config.json"));
        string schedulerPath = Path.Combine(path, "scheduler", "scheduler_config.json");
        var scheduler = File.Exists(schedulerPath) ? ReadJson(schedulerPath) : default;
        bool hasScheduler = scheduler.ValueKind == JsonValueKind.Object;
        if (hasScheduler && scheduler.String("prediction_type", "v_prediction") != "v_prediction")
        {
            throw new NotSupportedException("Stable Audio: only the v-objective scheduler of Stable Audio Open is supported.");
        }

        int headDim = transformer.Int("attention_head_dim", 64);
        var ratios = vae.Ints("downsampling_ratios");
        return new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [Prefix + "hidden_size"] = transformer.Int("num_attention_heads", 24) * headDim,
            [Prefix + "layer_count"] = transformer.Int("num_layers", 24),
            [Prefix + "head_count"] = transformer.Int("num_attention_heads", 24),
            [Prefix + "kv_head_count"] = transformer.Int("num_key_value_attention_heads", 12),
            [Prefix + "head_dim"] = headDim,
            [Prefix + "condition_width"] = transformer.Int("cross_attention_dim", 768),
            [Prefix + "latent_channels"] = transformer.Int("in_channels", 64),
            [Prefix + "frames"] = (int)transformer.Float("sample_size", 1024),
            [Prefix + "norm_eps"] = 1e-5f,
            [Prefix + "rope_dims"] = headDim / 2,
            [Prefix + "rope_base"] = 10000f,
            [Prefix + "timing.min_seconds"] = projection.Float("min_value", 0),
            [Prefix + "timing.max_seconds"] = projection.Float("max_value", 512),
            [Prefix + "sample_rate"] = vae.Int("sampling_rate", 44100),
            [Prefix + "audio_channels"] = vae.Int("audio_channels", 2),
            [Prefix + "samples_per_frame"] = ratios.Aggregate(1, (a, b) => a * b),
            [Prefix + "vae.upsampling_ratios"] = ratios.Reverse().Select(r => (long)r).ToArray(),
            [Prefix + "sigma_min"] = hasScheduler ? scheduler.Float("sigma_min", 0.3f) : 0.3f,
            [Prefix + "sigma_max"] = hasScheduler ? scheduler.Float("sigma_max", 500f) : 500f,
            [Prefix + "sigma_data"] = hasScheduler ? scheduler.Float("sigma_data", 1f) : 1f,
            [Prefix + "default_steps"] = 100,
            [Prefix + "default_guidance"] = 7f,
        };
    }

    private static int PromptLength(string tokenizer)
    {
        string config = Path.Combine(tokenizer, "tokenizer_config.json");
        return File.Exists(config) ? ReadJson(config).Int("model_max_length", 128) : 128;
    }

    private static JsonElement ReadJson(string file)
    {
        using var document = JsonDocument.Parse(File.ReadAllBytes(file));
        return document.RootElement.Clone();
    }
}
