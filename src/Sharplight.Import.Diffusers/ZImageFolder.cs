using System.Text.Json;
using Sharplight.Import.HuggingFace;
using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Reads a diffusers Z-Image pipeline folder (<c>transformer/</c> with a <c>ZImageTransformer2DModel</c>, <c>vae/</c>, <c>text_encoder/</c>
/// (Qwen3, opened by the Hugging Face importer), <c>tokenizer/</c>, <c>scheduler/</c>).
/// </summary>
internal static class ZImageFolder
{
    /// <summary>Whether the folder holds a Z-Image transformer.</summary>
    /// <param name="path">A directory.</param>
    public static bool IsFolder(string path)
    {
        string config = Path.Combine(path, "transformer", "config.json");
        if (!File.Exists(config))
        {
            return false;
        }

        using var document = JsonDocument.Parse(File.ReadAllBytes(config));
        return document.RootElement.String("_class_name", "") == "ZImageTransformer2DModel";
    }

    /// <summary>Opens the folder as a canonical source.</summary>
    /// <param name="path">The pipeline folder.</param>
    public static ModelSource Open(string path)
    {
        var tensors = new CheckpointTensors();
        ModelSource? textEncoder = null;
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

            ZImageNames.Add(transformer, "", weights);
            textEncoder = HuggingFaceImporter.Open(Path.Combine(path, "text_encoder"));
            string schedulerPath = Path.Combine(path, "scheduler", "scheduler_config.json");
            var config = ZImageConfig.FromWeights(weights.Weights)
                .WithDiffusers(File.Exists(schedulerPath) ? File.ReadAllText(schedulerPath) : null, ReadJson(path, "transformer"));
            var vae = VaeConfig.FromDiffusers(ReadJson(path, "vae"), FluxConfig.RgbFactors);
            string displayName = Path.GetFileName(Path.TrimEndingDirectorySeparator(Path.GetFullPath(path)));
            return ZImageSource.Create("diffusers", displayName, path, weights, config, tensors.Resources, vae, textEncoder);
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

    private static JsonElement ReadJson(string root, string component)
    {
        using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(root, component, "config.json")));
        return document.RootElement.Clone();
    }
}
