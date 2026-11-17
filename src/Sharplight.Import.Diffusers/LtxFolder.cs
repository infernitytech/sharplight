using System.Text.Json;
using Sharplight.ImageGeneration;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Reads a diffusers LTX-Video pipeline folder (<c>transformer/</c> with an <c>LTXVideoTransformer3DModel</c>, <c>vae/</c> with an
/// <c>AutoencoderKLLTXVideo</c> (non-causal decoder), <c>text_encoder/</c> (T5), <c>tokenizer/</c>, <c>scheduler/</c>) into the canonical
/// <c>ltx</c> source. A component folder without weights is left out (a folder that only holds the configurations and the T5 encoder
/// serves as a companion of a transformer-only file).
/// </summary>
internal static class LtxFolder
{
    /// <summary>The canonical architecture id (the <c>LtxAdapter</c>'s).</summary>
    public const string Architecture = "ltx";

    /// <summary>Metadata key prefix.</summary>
    public const string Prefix = Architecture + ".";
    private const string TextEncoderPrefix = "text_encoder.";

    /// <summary>The prompt length of the LTX pipeline.</summary>
    public const int PromptLength = 128;

    /// <summary>Whether the folder holds an LTX-Video transformer.</summary>
    /// <param name="path">A directory.</param>
    public static bool IsFolder(string path)
    {
        string config = Path.Combine(path, "transformer", "config.json");
        if (!File.Exists(config))
        {
            return false;
        }

        using var document = JsonDocument.Parse(File.ReadAllBytes(config));
        return document.RootElement.String("_class_name", "") == "LTXVideoTransformer3DModel";
    }

    /// <summary>Opens the folder as a canonical source.</summary>
    /// <param name="path">The pipeline folder.</param>
    public static ModelSource Open(string path)
    {
        var tensors = new CheckpointTensors();
        var resources = new List<IDisposable>();
        try
        {
            var transformerConfig = ReadJson(Path.Combine(path, "transformer", "config.json"));
            var vaeConfig = ReadJson(Path.Combine(path, "vae", "config.json"));
            if (vaeConfig.Bool("decoder_causal", false))
            {
                throw new NotSupportedException("Not supported yet: LTX-Video VAEs with a causal decoder; the non-causal decoders of LTX-Video 0.9.0 and 0.9.5 and later are supported.");
            }

            if (vaeConfig.TryGetProperty("decoder_inject_noise", out var inject) && inject.ValueKind == JsonValueKind.Array && inject.EnumerateArray().Any(e => e.GetBoolean()))
            {
                throw new NotSupportedException("Not supported yet: LTX-Video VAEs whose decoder injects noise (LTX-Video 0.9.1).");
            }

            foreach (string component in (string[])["transformer", "vae"])
            {
                if (HasWeights(path, component))
                {
                    DiffusersFolder.AddComponent(tensors, path, component);
                }
            }

            var weights = new CanonicalWeights();
            foreach (var (name, weight) in tensors.Tensors)
            {
                int dot = name.IndexOf('.');
                string component = name[..dot];
                string rest = name[(dot + 1)..];
                if (component == "transformer")
                {
                    AddTransformer(rest, weight, weights);
                }
                else if (rest.StartsWith("decoder.", StringComparison.Ordinal) || rest.StartsWith("latents_", StringComparison.Ordinal))
                {
                    AddVae(rest, weight, weights);
                }
            }

            resources.AddRange(tensors.Resources);
            var t5 = HasWeights(path, "text_encoder")
                ? FluxSource.AddT5Folder(Path.Combine(path, "text_encoder"), Path.Combine(path, "tokenizer"), weights, resources, TextEncoderPrefix)
                : new Dictionary<string, object>();
            int heads = transformerConfig.Int("num_attention_heads", 32);
            int[] scaling = vaeConfig.TryGetProperty("decoder_spatio_temporal_scaling", out var decoderScaling) && decoderScaling.ValueKind == JsonValueKind.Array
                ? [.. decoderScaling.EnumerateArray().Select(e => e.GetBoolean() ? 1 : 0)]
                : [1, 1, 1, 0];
            int upsamplers = scaling.Sum();
            int patch = vaeConfig.Int("patch_size", 4);
            var metadata = new Dictionary<string, object>(StringComparer.Ordinal)
            {
                [CanonicalKeys.Name] = Path.GetFileName(Path.TrimEndingDirectorySeparator(Path.GetFullPath(path))),
                [CanonicalKeys.Architecture] = Architecture,
                [Prefix + "transformer.latent_channels"] = transformerConfig.Int("in_channels", 128),
                [Prefix + "transformer.hidden_size"] = heads * transformerConfig.Int("attention_head_dim", 64),
                [Prefix + "transformer.head_count"] = heads,
                [Prefix + "transformer.layer_count"] = transformerConfig.Int("num_layers", 28),
                [Prefix + "transformer.caption_dim"] = transformerConfig.Int("caption_channels", 4096),
                [Prefix + "transformer.norm_eps"] = transformerConfig.Float("norm_eps", 1e-6f),
                [Prefix + "vae.latent_channels"] = vaeConfig.Int("latent_channels", 128),
                [Prefix + "vae.patch_size"] = patch,
                [Prefix + "vae.spatial_downscale"] = patch << upsamplers,
                [Prefix + "vae.temporal_downscale"] = 1 << upsamplers,
                // The decoder's blocks run in reverse order of the configuration lists.
                [Prefix + "vae.upsample_residual"] = vaeConfig.TryGetProperty("upsample_residual", out var residual) && residual.ValueKind == JsonValueKind.Array
                    ? residual.EnumerateArray().Select(e => e.GetBoolean()).Reverse().ToArray()
                    : Array.Empty<bool>(),
                [Prefix + "vae.upsample_factor"] = vaeConfig.TryGetProperty("upsample_factor", out var factor) && factor.ValueKind == JsonValueKind.Array
                    ? factor.EnumerateArray().Select(e => e.GetInt32()).Reverse().ToArray()
                    : Array.Empty<int>(),
                [Prefix + "vae.decode_timestep"] = 0f,
                [Prefix + "text_encoder.prompt_length"] = PromptLength,
                [Prefix + "default_width"] = 704,
                [Prefix + "default_height"] = 480,
                [Prefix + "default_frames"] = 161,
                [Prefix + "frames_per_second"] = 25.0,
                [Prefix + "default_steps"] = 50,
                [Prefix + "default_guidance"] = 3.0,
            };
            AddScheduler(path, metadata);
            const string encoder = "encoder_decoder.";
            foreach (var (key, value) in t5)
            {
                if (key.StartsWith(encoder, StringComparison.Ordinal))
                {
                    metadata[Prefix + "text_encoder." + key[encoder.Length..]] = value;
                }
                else if (key.StartsWith("tokenizer.", StringComparison.Ordinal))
                {
                    metadata[key] = value;
                }
            }

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
            tensors.DisposeAll();
            foreach (var resource in resources.Except(tensors.Resources))
            {
                resource.Dispose();
            }

            throw;
        }
    }

    /// <summary>The flow-matching Euler scheduler with the resolution-dependent shift and the terminal stretch.</summary>
    private static void AddScheduler(string path, Dictionary<string, object> metadata)
    {
        string s = Prefix + "scheduler.";
        string file = Path.Combine(path, "scheduler", "scheduler_config.json");
        var config = File.Exists(file) ? ReadJson(file) : default;
        bool present = File.Exists(file);
        metadata[s + "sampler"] = Samplers.Euler;
        metadata[s + "train_steps"] = present ? config.Int("num_train_timesteps", 1000) : 1000;
        metadata[s + "dynamic_shift"] = !present || config.Bool("use_dynamic_shifting", true);
        metadata[s + "base_shift"] = (double)(present ? config.Float("base_shift", 0.95f) : 0.95f);
        metadata[s + "max_shift"] = (double)(present ? config.Float("max_shift", 2.05f) : 2.05f);
        metadata[s + "base_sequence_length"] = present ? config.Int("base_image_seq_len", 1024) : 1024;
        metadata[s + "max_sequence_length"] = present ? config.Int("max_image_seq_len", 4096) : 4096;
        // LTX-Video 0.9.5 and later: a static shift and no terminal stretch (null in the configuration).
        metadata[s + "shift"] = present ? config.Float("shift", 1f) : 1f;
        metadata[s + "shift_terminal"] = !present ? 0.1f
            : config.TryGetProperty("shift_terminal", out var terminal) ? (terminal.ValueKind == JsonValueKind.Number ? terminal.GetSingle() : 0f) : 0.1f;
        if (present)
        {
            metadata[s + "source_config"] = File.ReadAllText(file);
        }
    }

    /// <summary>Adds a transformer tensor under its diffusers name (the modulation tables as f32 vectors).</summary>
    /// <param name="name">The diffusers name without <c>transformer.</c>.</param>
    /// <param name="stored">The stored tensor.</param>
    /// <param name="weights">The canonical set.</param>
    public static void AddTransformer(string name, WeightInfo stored, CanonicalWeights weights)
    {
        string canonical = "transformer." + name;
        if (name.EndsWith("scale_shift_table", StringComparison.Ordinal))
        {
            weights.AddF32(canonical, stored.As(canonical, new Shape(stored.Shape.Elements())));
            return;
        }

        weights.Add(canonical, stored);
    }

    /// <summary>Adds a VAE tensor under its diffusers name (3-D kernels with the time taps folded into the input channels).</summary>
    /// <param name="name">The diffusers name without <c>vae.</c>.</param>
    /// <param name="stored">The stored tensor.</param>
    /// <param name="weights">The canonical set.</param>
    public static void AddVae(string name, WeightInfo stored, CanonicalWeights weights)
    {
        string canonical = "vae." + name;
        if (name.StartsWith("latents_", StringComparison.Ordinal) || name.EndsWith("scale_shift_table", StringComparison.Ordinal)
            || name.EndsWith("timestep_scale_multiplier", StringComparison.Ordinal))
        {
            weights.AddF32(canonical, stored.Shape.Rank == 0 ? stored.As(canonical, new Shape(1)) : stored);
        }
        else if (stored.Shape.Rank == 5)
        {
            weights.Add(canonical, Conv3dKernels.FoldTimeTaps(canonical, stored));
        }
        else
        {
            weights.Add(canonical, stored);
        }
    }

    private static bool HasWeights(string path, string component)
        => Directory.Exists(Path.Combine(path, component)) && Directory.EnumerateFiles(Path.Combine(path, component), "*.safetensors").Any();

    private static JsonElement ReadJson(string file)
    {
        using var document = JsonDocument.Parse(File.ReadAllBytes(file));
        return document.RootElement.Clone();
    }
}
