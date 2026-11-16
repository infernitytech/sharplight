using System.Text.Json;
using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Reads a diffusers FLUX pipeline folder (<c>transformer/</c> with a <c>FluxTransformer2DModel</c>, <c>vae/</c>, <c>text_encoder/</c>
/// (CLIP), <c>text_encoder_2/</c> (T5), <c>tokenizer/</c>, <c>tokenizer_2/</c>, <c>scheduler/</c>).
/// </summary>
internal static class FluxFolder
{
    /// <summary>Whether the folder holds a FLUX transformer.</summary>
    /// <param name="path">A directory.</param>
    public static bool IsFolder(string path)
    {
        string config = Path.Combine(path, "transformer", "config.json");
        if (!File.Exists(config))
        {
            return false;
        }

        using var document = JsonDocument.Parse(File.ReadAllBytes(config));
        return document.RootElement.String("_class_name", "") == "FluxTransformer2DModel";
    }

    /// <summary>Opens the folder as a canonical source.</summary>
    /// <param name="path">The pipeline folder.</param>
    public static ModelSource Open(string path)
    {
        var tensors = new CheckpointTensors();
        var resources = new List<IDisposable>();
        try
        {
            DiffusersFolder.AddComponent(tensors, path, "transformer");
            DiffusersFolder.AddComponent(tensors, path, "vae");
            DiffusersFolder.AddComponent(tensors, path, "text_encoder");
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
                        DiffusersNames.AddVae(rest, weight, weights);
                        break;
                    default:
                        DiffusersNames.AddClipText(rest, weight, LatentDiffusionSource.TextEncoderPrefix, weights);
                        break;
                }
            }

            FluxNames.AddDiffusers(transformer, weights);
            resources.AddRange(tensors.Resources);
            var t5 = FluxSource.AddT5Folder(Path.Combine(path, "text_encoder_2"), Path.Combine(path, "tokenizer_2"), weights, resources);
            string schedulerPath = Path.Combine(path, "scheduler", "scheduler_config.json");
            var config = FluxConfig.FromWeights(weights.Weights)
                .WithDiffusers(File.Exists(schedulerPath) ? File.ReadAllText(schedulerPath) : null, ReadJson(path, "transformer"));
            var vae = VaeConfig.FromDiffusers(ReadJson(path, "vae"), FluxConfig.RgbFactors);
            var clip = ClipTextConfig.FromTransformers(ReadJson(path, "text_encoder"));
            var vocabulary = ClipVocabulary.FromFolder(Path.Combine(path, "tokenizer"));
            string displayName = Path.GetFileName(Path.TrimEndingDirectorySeparator(Path.GetFullPath(path)));
            return FluxSource.Create("diffusers", displayName, path, weights, config, resources, vae, clip, vocabulary, t5);
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

    private static JsonElement ReadJson(string root, string component)
    {
        using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(root, component, "config.json")));
        return document.RootElement.Clone();
    }
}
