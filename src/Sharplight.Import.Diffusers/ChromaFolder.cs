using System.Text.Json;
using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Reads a diffusers Chroma pipeline folder (<c>transformer/</c> with a <c>ChromaTransformer2DModel</c>, <c>vae/</c>, <c>text_encoder/</c>
/// (T5), <c>tokenizer/</c>, <c>scheduler/</c>).
/// </summary>
internal static class ChromaFolder
{
    /// <summary>Whether the folder holds a Chroma transformer.</summary>
    /// <param name="path">A directory.</param>
    public static bool IsFolder(string path)
    {
        string config = Path.Combine(path, "transformer", "config.json");
        if (!File.Exists(config))
        {
            return false;
        }

        using var document = JsonDocument.Parse(File.ReadAllBytes(config));
        return document.RootElement.String("_class_name", "") == "ChromaTransformer2DModel";
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
            var weights = new CanonicalWeights();
            var transformer = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
            foreach (var (name, weight) in tensors.Tensors)
            {
                int dot = name.IndexOf('.');
                string rest = name[(dot + 1)..];
                if (name[..dot] == "transformer")
                {
                    transformer[rest] = weight;
                }
                else
                {
                    DiffusersNames.AddVae(rest, weight, weights);
                }
            }

            FluxNames.AddDiffusers(transformer, weights);
            resources.AddRange(tensors.Resources);
            var t5 = FluxSource.AddT5Folder(Path.Combine(path, "text_encoder"), Path.Combine(path, "tokenizer"), weights, resources, ChromaSource.T5Prefix);
            string schedulerPath = Path.Combine(path, "scheduler", "scheduler_config.json");
            var config = ChromaConfig.FromWeights(weights.Weights)
                .WithDiffusers(File.Exists(schedulerPath) ? File.ReadAllText(schedulerPath) : null, ReadJson(path, "transformer"));
            var vae = VaeConfig.FromDiffusers(ReadJson(path, "vae"), FluxConfig.RgbFactors);
            string displayName = Path.GetFileName(Path.TrimEndingDirectorySeparator(Path.GetFullPath(path)));
            return ChromaSource.Create("diffusers", displayName, path, weights, config, resources, vae, t5);
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
