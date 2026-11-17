using System.Text.Json;
using Sharplight.Import.HuggingFace;
using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Reads a diffusers Wan pipeline folder (<c>transformer/</c> with a <c>WanTransformer3DModel</c>, <c>vae/</c> with an
/// <c>AutoencoderKLWan</c>, <c>text_encoder/</c> (umT5), <c>tokenizer/</c>, <c>scheduler/</c>, <c>model_index.json</c>; Wan 2.1 I2V /
/// FLF2V also <c>image_encoder/</c> (CLIP ViT-H/14) and <c>image_processor/</c>) into the canonical <c>wan</c> source. A component folder
/// without weights is left out (a folder that only holds the configurations and the image encoder serves as a companion of a
/// transformer-only file).
/// </summary>
internal static class WanFolder
{
    /// <summary>Canonical weight prefix of the umT5 encoder.</summary>
    public const string TextEncoderPrefix = "text_encoder.";

    /// <summary>The prompt length of the Wan pipelines.</summary>
    private const int PromptLength = 512;

    /// <summary>Whether the folder holds a Wan transformer.</summary>
    /// <param name="path">A directory.</param>
    public static bool IsFolder(string path)
    {
        string config = Path.Combine(path, "transformer", "config.json");
        if (!File.Exists(config))
        {
            return false;
        }

        using var document = JsonDocument.Parse(File.ReadAllBytes(config));
        return document.RootElement.String("_class_name", "") == "WanTransformer3DModel";
    }

    /// <summary>Opens the folder as a canonical source.</summary>
    /// <param name="path">The pipeline folder.</param>
    public static ModelSource Open(string path)
    {
        var tensors = new CheckpointTensors();
        var resources = new List<IDisposable>();
        try
        {
            bool twoExperts = Directory.Exists(Path.Combine(path, "transformer_2"));
            foreach (string component in (string[])["transformer", "transformer_2", "vae", "image_encoder"])
            {
                if (HasWeights(path, component))
                {
                    DiffusersFolder.AddComponent(tensors, path, component);
                }
            }

            var imageEncoder = HasWeights(path, "image_encoder") ? ReadJson(Path.Combine(path, "image_encoder", "config.json")) : (JsonElement?)null;
            // Wan conditions on the penultimate hidden states: the layers before the last one.
            int imageLayers = imageEncoder is { } clip ? clip.Int("num_hidden_layers", 32) - 1 : 0;
            var weights = new CanonicalWeights();
            foreach (var (name, weight) in tensors.Tensors)
            {
                int dot = name.IndexOf('.');
                string component = name[..dot];
                string rest = name[(dot + 1)..];
                switch (component)
                {
                    case "transformer":
                        WanNames.AddTransformer(rest, weight, weights);
                        break;
                    case "transformer_2":
                        WanNames.AddTransformer(rest, weight, weights, WanNames.SecondTransformerPrefix);
                        break;
                    case "image_encoder":
                        WanNames.AddImageEncoder(rest, weight, weights, imageLayers);
                        break;
                    default:
                        WanNames.AddVae(rest, weight, weights);
                        break;
                }
            }

            resources.AddRange(tensors.Resources);
            var t5 = HasWeights(path, "text_encoder")
                ? FluxSource.AddT5Folder(Path.Combine(path, "text_encoder"), Path.Combine(path, "tokenizer"), weights, resources, TextEncoderPrefix)
                : new Dictionary<string, object>();
            string schedulerPath = Path.Combine(path, "scheduler", "scheduler_config.json");
            string indexPath = Path.Combine(path, "model_index.json");
            var config = WanConfig.FromDiffusers(ReadJson(Path.Combine(path, "transformer", "config.json")), ReadJson(Path.Combine(path, "vae", "config.json")),
                File.Exists(schedulerPath) ? File.ReadAllText(schedulerPath) : null, File.Exists(indexPath) ? ReadJson(indexPath) : null);
            var metadata = new Dictionary<string, object>(config.Metadata, StringComparer.Ordinal)
            {
                [CanonicalKeys.Name] = Path.GetFileName(Path.TrimEndingDirectorySeparator(Path.GetFullPath(path))),
                [CanonicalKeys.Architecture] = WanConfig.Architecture,
                [WanConfig.Prefix + "text_encoder.prompt_length"] = PromptLength,
            };
            if (imageEncoder is { } vision)
            {
                AddImageEncoderMetadata(metadata, vision, imageLayers, path);
            }

            const string encoder = "encoder_decoder.";
            foreach (var (key, value) in t5)
            {
                if (key.StartsWith(encoder, StringComparison.Ordinal))
                {
                    metadata[WanConfig.Prefix + "text_encoder." + key[encoder.Length..]] = value;
                }
                else if (key.StartsWith("tokenizer.", StringComparison.Ordinal))
                {
                    metadata[key] = value;
                }
            }

            return new ModelSource
            {
                Format = "diffusers",
                Architecture = WanConfig.Architecture,
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

    /// <summary>The image tower keys (<c>wan.image_encoder.*</c>) and its image processor.</summary>
    private static void AddImageEncoderMetadata(Dictionary<string, object> metadata, JsonElement vision, int layers, string path)
    {
        const string e = WanConfig.Prefix + "image_encoder.";
        int imageSize = vision.Int("image_size", 224);
        string act = vision.String("hidden_act", "quick_gelu");
        metadata[e + "layer_count"] = layers;
        metadata[e + "hidden_size"] = vision.Int("hidden_size", 1280);
        metadata[e + "head_count"] = vision.Int("num_attention_heads", 16);
        metadata[e + "norm_eps"] = vision.Float("layer_norm_eps", 1e-5f);
        metadata[e + "activation"] = act == "quick_gelu" ? "quick_gelu" : "gelu";
        metadata[e + "patch_size"] = vision.Int("patch_size", 14);
        metadata[e + "image_size"] = imageSize;
        var processor = HfImageProcessorConfig.Read(Path.Combine(path, "image_processor"), imageSize);
        metadata[e + "processor"] = processor.ToJson().ToJsonString();
    }

    private static bool HasWeights(string path, string component)
        => Directory.Exists(Path.Combine(path, component)) && Directory.EnumerateFiles(Path.Combine(path, component), "*.safetensors").Any();

    private static JsonElement ReadJson(string file)
    {
        using var document = JsonDocument.Parse(File.ReadAllBytes(file));
        return document.RootElement.Clone();
    }
}
