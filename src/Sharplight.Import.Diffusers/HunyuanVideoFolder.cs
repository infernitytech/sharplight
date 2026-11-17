using System.Text.Json;
using Sharplight.Import.HuggingFace;
using Sharplight.ImageGeneration;
using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Reads a diffusers HunyuanVideo pipeline folder (<c>transformer/</c> with a <c>HunyuanVideoTransformer3DModel</c>, <c>vae/</c> with an
/// <c>AutoencoderKLHunyuanVideo</c>, <c>text_encoder/</c> (a Llama model) with <c>tokenizer/</c>, <c>text_encoder_2/</c> (CLIP-L) with
/// <c>tokenizer_2/</c>, <c>scheduler/</c>) into the canonical <c>hunyuan_video</c> source: the transformer under the original names
/// (<see cref="HunyuanVideoNames"/>), the VAE under diffusers' names (its 3-D kernels with the time taps folded into the input channels), the language model as an embedded canonical source,
/// CLIP in the image generators' CLIP vocabulary. A component folder without weights is left out (a folder that only holds the
/// configurations and the encoders serves as a companion of a transformer-only file).
/// </summary>
internal static class HunyuanVideoFolder
{
    /// <summary>The canonical architecture id (the <c>HunyuanVideoAdapter</c>'s).</summary>
    public const string Architecture = "hunyuan_video";

    /// <summary>Metadata key prefix.</summary>
    public const string Prefix = Architecture + ".";

    /// <summary>The prompt template of the HunyuanVideo pipeline (its system instruction), with the BOS token and <c>{prompt}</c>.</summary>
    public const string PromptTemplate = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
        + "1. The main content and theme of the video."
        + "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
        + "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
        + "4. background environment, light, style and atmosphere."
        + "5. camera angles, movements, and transitions used in the video:<|eot_id|>"
        + "<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>";

    private const string TextEncoderSourcePrefix = Prefix + "text_encoder_source.";
    private const string TextEncoderPrefix = "text_encoder.";
    private const string SecondTextEncoderPrefix = "text_encoder_2.";

    /// <summary>Whether the folder holds a HunyuanVideo transformer configuration.</summary>
    /// <param name="path">A directory.</param>
    public static bool IsFolder(string path)
    {
        string config = Path.Combine(path, "transformer", "config.json");
        if (!File.Exists(config))
        {
            return false;
        }

        using var document = JsonDocument.Parse(File.ReadAllBytes(config));
        return document.RootElement.String("_class_name", "") == "HunyuanVideoTransformer3DModel";
    }

    /// <summary>Opens the folder as a canonical source.</summary>
    /// <param name="path">The pipeline folder.</param>
    public static ModelSource Open(string path)
    {
        var tensors = new CheckpointTensors();
        var resources = new List<IDisposable>();
        ModelSource? textEncoder = null;
        try
        {
            foreach (string component in (string[])["transformer", "vae", "text_encoder_2"])
            {
                if (HasWeights(path, component))
                {
                    DiffusersFolder.AddComponent(tensors, path, component);
                }
            }

            var weights = new CanonicalWeights();
            var transformer = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
            foreach (var (name, weight) in tensors.Tensors)
            {
                int dot = name.IndexOf('.');
                string component = name[..dot];
                string rest = name[(dot + 1)..];
                switch (component)
                {
                    case "transformer":
                        transformer[rest] = weight;
                        break;
                    case "vae":
                        AddVae(rest, weight, weights);
                        break;
                    default:
                        DiffusersNames.AddClipText(rest, weight, SecondTextEncoderPrefix, weights);
                        break;
                }
            }

            HunyuanVideoNames.AddDiffusers(transformer, weights);
            resources.AddRange(tensors.Resources);
            var metadata = new Dictionary<string, object>(StringComparer.Ordinal)
            {
                [CanonicalKeys.Name] = Path.GetFileName(Path.TrimEndingDirectorySeparator(Path.GetFullPath(path))),
                [CanonicalKeys.Architecture] = Architecture,
            };
            var allWeights = new Dictionary<string, WeightInfo>(weights.Weights, StringComparer.Ordinal);
            if (HasWeights(path, "text_encoder"))
            {
                textEncoder = HuggingFaceImporter.Open(Path.Combine(path, "text_encoder"));
                foreach (var (key, value) in textEncoder.Metadata)
                {
                    metadata[TextEncoderSourcePrefix + key] = value;
                    if (key.StartsWith("tokenizer.", StringComparison.Ordinal))
                    {
                        metadata[key] = value;
                    }
                }

                metadata[TextEncoderSourcePrefix + "architecture"] = textEncoder.Architecture;
                foreach (var (name, weight) in textEncoder.Weights)
                {
                    allWeights[TextEncoderPrefix + name] = weight.As(TextEncoderPrefix + name);
                }

                resources.AddRange(textEncoder.OwnedResources);
            }

            if (HasWeights(path, "text_encoder_2"))
            {
                var clip = ClipTextConfig.FromTransformers(ReadJson(Path.Combine(path, "text_encoder_2", "config.json")));
                clip.WriteTo(metadata, Prefix, "text_encoder_2");
                ClipVocabulary.FromFolder(Path.Combine(path, "tokenizer_2")).WriteTo(metadata, Prefix + "tokenizer_2.", clip.ContextLength);
            }

            if (File.Exists(Path.Combine(path, "vae", "config.json")))
            {
                AddVaeConfig(metadata, ReadJson(Path.Combine(path, "vae", "config.json")));
            }

            AddTransformerConfig(metadata, ReadJson(Path.Combine(path, "transformer", "config.json")));
            string schedulerPath = Path.Combine(path, "scheduler", "scheduler_config.json");
            AddDefaults(metadata, File.Exists(schedulerPath) ? File.ReadAllText(schedulerPath) : null);
            return new ModelSource
            {
                Format = "diffusers",
                Architecture = Architecture,
                Metadata = metadata,
                Weights = allWeights,
                OwnedResources = resources,
                Path = path,
            };
        }
        catch
        {
            tensors.DisposeAll();
            foreach (var resource in textEncoder?.OwnedResources ?? [])
            {
                resource.Dispose();
            }

            throw;
        }
    }

    /// <summary>A VAE tensor under its diffusers name; 3-D kernels with the time taps folded into the input channels.</summary>
    private static void AddVae(string name, WeightInfo stored, CanonicalWeights weights)
    {
        string canonical = "vae." + name;
        if (stored.Shape.Rank == 5)
        {
            weights.Add(canonical, Conv3dKernels.FoldTimeTaps(canonical, stored));
        }
        else if (name.Contains(".attentions.", StringComparison.Ordinal) && stored.Shape.Rank == 4)
        {
            weights.AddLinear(canonical, stored);
        }
        else
        {
            weights.Add(canonical, stored);
        }
    }

    private static void AddTransformerConfig(Dictionary<string, object> metadata, JsonElement config)
    {
        string t = Prefix + "transformer.";
        int inChannels = config.Int("in_channels", 16);
        metadata[t + "in_channels"] = inChannels;
        metadata[t + "out_channels"] = config.Int("out_channels", inChannels);
        metadata[t + "patch_size"] = config.Int("patch_size", 2);
        metadata[t + "patch_size_t"] = config.Int("patch_size_t", 1);
        metadata[t + "head_count"] = config.Int("num_attention_heads", 24);
        metadata[t + "hidden_size"] = config.Int("num_attention_heads", 24) * config.Int("attention_head_dim", 128);
        metadata[t + "head_dim"] = config.Int("attention_head_dim", 128);
        metadata[t + "double_layers"] = config.Int("num_layers", 20);
        metadata[t + "single_layers"] = config.Int("num_single_layers", 40);
        metadata[t + "refiner_layers"] = config.Int("num_refiner_layers", 2);
        metadata[t + "text_dim"] = config.Int("text_embed_dim", 4096);
        metadata[t + "pooled_dim"] = config.Int("pooled_projection_dim", 768);
        metadata[t + "guidance_embed"] = config.Bool("guidance_embeds", true);
        metadata[t + "rope_theta"] = config.Float("rope_theta", 256f);
        metadata[t + "rope_axes_dim"] = config.TryGetProperty("rope_axes_dim", out _) ? config.Ints("rope_axes_dim") : new[] { 16, 56, 56 };
        if (config.TryGetProperty("image_condition_type", out var condition) && condition.ValueKind == JsonValueKind.String)
        {
            throw new NotSupportedException($"Not supported yet: HunyuanVideo image-to-video ({condition.GetString()} conditioning); HunyuanVideo text-to-video is supported.");
        }
    }

    private static void AddVaeConfig(Dictionary<string, object> metadata, JsonElement config)
    {
        string v = Prefix + "vae.";
        metadata[v + "latent_channels"] = config.Int("latent_channels", 16);
        metadata[v + "scaling_factor"] = config.Float("scaling_factor", 0.476986f);
        metadata[v + "groups"] = config.Int("norm_num_groups", 32);
        metadata[v + "temporal_compression"] = config.Int("temporal_compression_ratio", 4);
        metadata[v + "spatial_compression"] = config.Int("spatial_compression_ratio", 8);
        // diffusers decodes more than 4 latent frames in tiles of 5, 3 apart (tile_sample_min_num_frames 16, stride 12).
        metadata[v + "tile_frames"] = 4;
        metadata[v + "tile_stride"] = 3;
    }

    /// <summary>The prompt handling, scheduler and pipeline defaults (720×1280, 129 frames at 24 fps, 50 steps, embedded guidance 6).</summary>
    /// <param name="metadata">The canonical metadata.</param>
    /// <param name="scheduler">The scheduler's <c>scheduler_config.json</c> text, or null.</param>
    public static void AddDefaults(Dictionary<string, object> metadata, string? scheduler)
    {
        metadata[Prefix + "text_encoder.prompt_length"] = 256;
        metadata[Prefix + "text_encoder.prompt_template"] = PromptTemplate;
        metadata[Prefix + "text_encoder.prompt_prefix"] = 95;
        string s = Prefix + "scheduler.";
        float shift = 7f;
        int trainSteps = 1000;
        if (scheduler is not null)
        {
            using var document = JsonDocument.Parse(scheduler);
            shift = document.RootElement.Float("shift", shift);
            trainSteps = document.RootElement.Int("num_train_timesteps", trainSteps);
            metadata[s + "source_config"] = scheduler;
        }

        metadata[s + "shift"] = shift;
        metadata[s + "train_steps"] = trainSteps;
        metadata[s + "sampler"] = Samplers.Euler;
        metadata[Prefix + "default_width"] = 1280;
        metadata[Prefix + "default_height"] = 720;
        metadata[Prefix + "default_frames"] = 129;
        metadata[Prefix + "frames_per_second"] = 24.0;
        metadata[Prefix + "default_steps"] = 50;
        metadata[Prefix + "default_guidance"] = 6.0;
    }

    private static bool HasWeights(string path, string component)
        => Directory.Exists(Path.Combine(path, component)) && Directory.EnumerateFiles(Path.Combine(path, component), "*.safetensors").Any();

    private static JsonElement ReadJson(string file)
    {
        using var document = JsonDocument.Parse(File.ReadAllBytes(file));
        return document.RootElement.Clone();
    }
}
