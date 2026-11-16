using System.Globalization;
using System.Text.Json;
using Sharplight.Import.HuggingFace;
using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Stand-alone component files of diffusion pipelines, as they are distributed next to transformer-only checkpoints: a CLIP text
/// encoder (<c>clip_l.safetensors</c>, Hugging Face names), a KL autoencoder (<c>ae.safetensors</c>, original or diffusers names) and a
/// T5 encoder (<c>t5xxl_*.safetensors</c>, Hugging Face names; T5 encoder GGUF files are read by the GGUF importer). They open as the
/// component sources the image-generation adapters merge (architectures <c>clip_text_encoder</c>, <c>kl_autoencoder</c> and
/// <c>encoder_decoder</c>). A <c>config.json</c> beside the file (a diffusers component folder) supplies the configuration; otherwise it
/// is inferred from the shapes with the published models' constants.
/// </summary>
internal static class ComponentCheckpoint
{
    /// <summary>Canonical architecture of a CLIP text encoder source.</summary>
    public const string ClipArchitecture = "clip_text_encoder";

    /// <summary>Canonical architecture of a KL autoencoder source.</summary>
    public const string AutoencoderArchitecture = "kl_autoencoder";

    /// <summary>Canonical architecture of a T5 encoder source.</summary>
    public const string T5Architecture = "encoder_decoder";

    /// <summary>The kind of component a file holds, from its tensor names, or null.</summary>
    /// <param name="names">The tensor names.</param>
    public static string? Classify(IReadOnlyCollection<string> names)
    {
        bool Has(string name) => names.Contains(name);
        if (names.Any(n => n.StartsWith("vision_model.", StringComparison.Ordinal) || n.StartsWith("model.diffusion_model.", StringComparison.Ordinal)))
        {
            return null;
        }

        if (Has("text_model.encoder.layers.0.self_attn.q_proj.weight") || Has("encoder.layers.0.self_attn.q_proj.weight") && Has("embeddings.token_embedding.weight"))
        {
            return ClipArchitecture;
        }

        if ((Has("encoder.conv_in.weight") || Has("encoder.down_blocks.0.resnets.0.conv1.weight")) && Has("decoder.conv_in.weight"))
        {
            return AutoencoderArchitecture;
        }

        if (Has("encoder.block.0.layer.0.SelfAttention.q.weight") && !names.Any(n => n.StartsWith("decoder.block.", StringComparison.Ordinal)))
        {
            return T5Architecture;
        }

        return null;
    }

    /// <summary>The component kind of a safetensors file, or null (also for unreadable files).</summary>
    /// <param name="path">A file.</param>
    public static string? ClassifyFile(string path)
    {
        if (!path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase) || !File.Exists(path))
        {
            return null;
        }

        try
        {
            using var file = SafetensorsFile.Open(path);
            return Classify([.. file.Tensors.Keys]);
        }
        catch (IOException)
        {
            return null;
        }
        catch (InvalidDataException)
        {
            return null;
        }
    }

    /// <summary>Opens a component file.</summary>
    /// <param name="path">The safetensors file.</param>
    /// <exception cref="NotSupportedException">The file is not a known component.</exception>
    public static ModelSource Open(string path)
    {
        var tensors = new CheckpointTensors();
        try
        {
            tensors.Add(path);
            string kind = Classify([.. tensors.Tensors.Keys]) ?? throw new NotSupportedException($"'{path}' is not a CLIP text encoder, KL autoencoder or T5 encoder.");
            string? configPath = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(path))!, "config.json");
            JsonElement? config = File.Exists(configPath) ? ReadJson(configPath) : null;
            var metadata = new Dictionary<string, object>(StringComparer.Ordinal)
            {
                [CanonicalKeys.Name] = Path.GetFileNameWithoutExtension(path),
                [CanonicalKeys.Architecture] = kind,
            };
            var weights = new CanonicalWeights();
            switch (kind)
            {
                case ClipArchitecture:
                    OpenClip(path, tensors, config, weights, metadata);
                    break;
                case AutoencoderArchitecture:
                    OpenVae(tensors, config, weights, metadata);
                    break;
                default:
                    return OpenT5(path, tensors, config, metadata);
            }

            return new ModelSource
            {
                Format = "safetensors",
                Architecture = kind,
                Metadata = metadata,
                Weights = weights.Weights,
                OwnedResources = tensors.Resources,
                Path = path,
            };
        }
        catch
        {
            tensors.DisposeAll();
            throw;
        }
    }

    private static void OpenClip(string path, CheckpointTensors tensors, JsonElement? config, CanonicalWeights weights, Dictionary<string, object> metadata)
    {
        foreach (var (name, weight) in tensors.Tensors)
        {
            DiffusersNames.AddClipText(name, weight, "", weights);
        }

        // Without a configuration: OpenAI CLIP-L uses quick-GELU, OpenCLIP bigG (SDXL's / SD3's second encoder, 1280 wide) GELU.
        var clip = config is { } c && c.String("model_type", "") is "clip_text_model" or "clip"
            ? ClipTextConfig.FromTransformers(c)
            : ClipTextConfig.FromWeights(weights.Weights, "", weights.Weights["token_embedding.weight"].Shape[1].Fixed == 1280 ? "gelu" : "quick_gelu");
        clip.WriteTo(metadata, "", ClipArchitecture);
        // A diffusers text_encoder folder has its tokenizer beside it; stand-alone files use the OpenAI CLIP vocabulary.
        string sibling = Path.Combine(Path.GetDirectoryName(Path.GetDirectoryName(Path.GetFullPath(path)))!, "tokenizer");
        var vocabulary = config is not null && Directory.Exists(sibling) ? ClipVocabulary.FromFolder(sibling) : ClipVocabulary.OpenAi();
        vocabulary.WriteTo(metadata, "", clip.ContextLength);
    }

    private static void OpenVae(CheckpointTensors tensors, JsonElement? config, CanonicalWeights weights, Dictionary<string, object> metadata)
    {
        var names = tensors.Tensors.Keys.ToList();
        if (names.Contains("encoder.conv_in.weight"))
        {
            foreach (var (ldm, diffusers) in LdmNames.Vae(names))
            {
                DiffusersNames.AddVae(diffusers, tensors.Tensors[ldm], weights);
            }
        }
        else
        {
            foreach (var (name, weight) in tensors.Tensors)
            {
                DiffusersNames.AddVae(name, weight, weights);
            }
        }

        int latentChannels = (int)weights.Weights["vae.decoder.input_conv.weight"].Shape[1].Fixed;
        bool flux = latentChannels == 16;
        var vae = config is { } c && c.String("_class_name", "") == "AutoencoderKL"
            ? VaeConfig.FromDiffusers(c, flux ? FluxConfig.RgbFactors : LatentDiffusionSource.StableDiffusion1RgbFactors)
            : VaeConfig.FromWeights(weights.Weights, flux ? FluxConfig.VaeScalingFactor : 0.18215f, flux ? FluxConfig.RgbFactors : LatentDiffusionSource.StableDiffusion1RgbFactors)
                with { ShiftFactor = flux ? FluxConfig.VaeShiftFactor : 0f };
        vae.WriteTo(metadata, AutoencoderArchitecture + ".");
    }

    /// <summary>
    /// A T5 encoder from Hugging Face names: the configuration from <c>config.json</c> beside the file or inferred (T5 v1.1: gated
    /// tanh-GELU feed-forward, ε 1e-6), the tokenizer from a <c>tokenizer.json</c> beside the file or in the sibling tokenizer folder of
    /// a diffusers layout (<c>text_encoder_2</c> → <c>tokenizer_2</c>).
    /// </summary>
    private static ModelSource OpenT5(string path, CheckpointTensors tensors, JsonElement? config, Dictionary<string, object> metadata)
    {
        string folder = Path.GetDirectoryName(Path.GetFullPath(path))!;
        IReadOnlyDictionary<string, SafetensorsTensor> stored = ((SafetensorsFile)tensors.Resources.Single()).Tensors;
        IReadOnlyList<IDisposable> resources = tensors.Resources;
        if (IsShard(path))
        {
            // One shard of a sharded encoder (a diffusers text_encoder_3 folder): the whole checkpoint.
            tensors.DisposeAll();
            var checkpoint = SafetensorsCheckpoint.Open(folder);
            stored = checkpoint.Tensors;
            resources = [checkpoint];
        }

        var hfConfig = config is { } c && c.String("model_type", "") is "t5" or "umt5" or "mt5"
            ? HfConfig.Load(folder)
            : HfConfig.Parse(InferT5Config(stored), folder);
        var options = new HuggingFaceImportOptions();
        var mapping = new T5HfMapping();
        var weights = new CanonicalWeights();
        foreach (var (name, weight) in mapping.MapWeights(hfConfig, stored, options))
        {
            weights.Add(name, weight);
        }

        foreach (var (key, value) in mapping.MapHyperParameters(hfConfig, options))
        {
            metadata[key] = value;
        }

        // A diffusers component folder text_encoder{suffix} has its tokenizer in tokenizer{suffix} (FLUX: _2, SD3: _3, Chroma: none).
        string folderName = Path.GetFileName(folder);
        string? sibling = folderName.StartsWith("text_encoder", StringComparison.Ordinal)
            ? Path.Combine(Path.GetDirectoryName(folder) ?? folder, "tokenizer" + folderName["text_encoder".Length..])
            : null;
        string? tokenizer = new[] { folder, sibling }.OfType<string>().FirstOrDefault(d => File.Exists(Path.Combine(d, "tokenizer.json")));
        if (tokenizer is not null)
        {
            HfTokenizerTranslator.AddTo(metadata, tokenizer, hfConfig);
        }
        else
        {
            T5Vocabulary.WriteTo(metadata, hfConfig);
        }

        return new ModelSource
        {
            Format = "safetensors",
            Architecture = T5Architecture,
            Metadata = metadata,
            Weights = weights.Weights,
            OwnedResources = resources,
            Path = path,
        };
    }

    /// <summary>Whether the file is a shard of a checkpoint with a <c>*.safetensors.index.json</c> beside it.</summary>
    private static bool IsShard(string path)
        => System.Text.RegularExpressions.Regex.IsMatch(Path.GetFileName(path), @"-\d{5}-of-\d{5}\.safetensors$")
           && Directory.EnumerateFiles(Path.GetDirectoryName(Path.GetFullPath(path))!, "*.safetensors.index.json").Any();

    /// <summary>A T5 v1.1 encoder configuration (JSON) from the tensor shapes.</summary>
    private static string InferT5Config(IReadOnlyDictionary<string, SafetensorsTensor> tensors)
    {
        int layers = 0;
        while (tensors.ContainsKey(string.Create(CultureInfo.InvariantCulture, $"encoder.block.{layers}.layer.0.SelfAttention.q.weight")))
        {
            layers++;
        }

        var q = tensors["encoder.block.0.layer.0.SelfAttention.q.weight"].Shape;
        var bias = tensors["encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight"].Shape;
        var up = tensors.TryGetValue("encoder.block.0.layer.1.DenseReluDense.wi_0.weight", out var gate) ? gate.Shape : tensors["encoder.block.0.layer.1.DenseReluDense.wi.weight"].Shape;
        long heads = bias[1];
        bool gated = tensors.ContainsKey("encoder.block.0.layer.1.DenseReluDense.wi_0.weight");
        return string.Create(CultureInfo.InvariantCulture,
            $$"""{"model_type": "t5", "architectures": ["T5EncoderModel"], "d_model": {{q[1]}}, "num_layers": {{layers}}, "num_heads": {{heads}}, "d_kv": {{q[0] / heads}}, "d_ff": {{up[0]}}, "feed_forward_proj": "{{(gated ? "gated-gelu" : "relu")}}", "layer_norm_epsilon": 1e-6, "relative_attention_num_buckets": {{bias[0]}}, "relative_attention_max_distance": 128, "vocab_size": 32128, "eos_token_id": 1, "pad_token_id": 0}""");
    }

    private static JsonElement ReadJson(string path)
    {
        using var document = JsonDocument.Parse(File.ReadAllBytes(path));
        return document.RootElement.Clone();
    }
}
