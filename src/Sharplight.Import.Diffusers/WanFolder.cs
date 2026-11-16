using System.Text.Json;
using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Reads a diffusers Wan pipeline folder (<c>transformer/</c> with a <c>WanTransformer3DModel</c>, <c>vae/</c> with an
/// <c>AutoencoderKLWan</c>, <c>text_encoder/</c> (umT5), <c>tokenizer/</c>, <c>scheduler/</c>, <c>model_index.json</c>) into the
/// canonical <c>wan</c> source.
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
            if (Directory.Exists(Path.Combine(path, "transformer_2")))
            {
                throw new NotSupportedException("Not supported yet: Wan 2.2 mixture-of-experts pipelines (A14B, two transformers); Wan 2.1 and Wan 2.2 TI2V-5B are supported.");
            }

            DiffusersFolder.AddComponent(tensors, path, "transformer");
            DiffusersFolder.AddComponent(tensors, path, "vae");
            var weights = new CanonicalWeights();
            foreach (var (name, weight) in tensors.Tensors)
            {
                int dot = name.IndexOf('.');
                string component = name[..dot];
                string rest = name[(dot + 1)..];
                if (component == "transformer")
                {
                    WanNames.AddTransformer(rest, weight, weights);
                }
                else
                {
                    WanNames.AddVae(rest, weight, weights);
                }
            }

            resources.AddRange(tensors.Resources);
            var t5 = FluxSource.AddT5Folder(Path.Combine(path, "text_encoder"), Path.Combine(path, "tokenizer"), weights, resources, TextEncoderPrefix);
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

    private static JsonElement ReadJson(string file)
    {
        using var document = JsonDocument.Parse(File.ReadAllBytes(file));
        return document.RootElement.Clone();
    }
}
