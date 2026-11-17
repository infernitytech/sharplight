using System.Text.Json;
using Sharplight.Import.HuggingFace;
using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Reads a diffusers Krea 2 pipeline folder (<c>transformer/</c> with a <c>Krea2Transformer2DModel</c>, <c>vae/</c> with the Qwen-Image
/// <c>AutoencoderKLQwenImage</c> (the Wan 2.1 VAE), <c>text_encoder/</c> (Qwen3-VL-4B, opened by the Hugging Face importer; its vision tower is
/// unused), <c>tokenizer/</c>, <c>scheduler/</c>, <c>model_index.json</c> with the tapped layers and <c>is_distilled</c>).
/// </summary>
internal static class Krea2Folder
{
    /// <summary>Whether the folder holds a Krea 2 transformer.</summary>
    /// <param name="path">A directory.</param>
    public static bool IsFolder(string path)
    {
        string config = Path.Combine(path, "transformer", "config.json");
        if (!File.Exists(config))
        {
            return false;
        }

        using var document = JsonDocument.Parse(File.ReadAllBytes(config));
        return document.RootElement.String("_class_name", "") == "Krea2Transformer2DModel";
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
                    WanNames.AddVae(rest, weight, weights);
                }
            }

            Krea2Names.Add(transformer, "", weights);
            textEncoder = HuggingFaceImporter.Open(Path.Combine(path, "text_encoder"));
            string schedulerPath = Path.Combine(path, "scheduler", "scheduler_config.json");
            JsonElement? index = File.Exists(Path.Combine(path, "model_index.json")) ? ReadJson(path, "", "model_index.json") : null;
            var config = Krea2Config.FromWeights(weights.Weights)
                .WithDiffusers(File.Exists(schedulerPath) ? File.ReadAllText(schedulerPath) : null, ReadJson(path, "transformer"), index);
            var vae = new Dictionary<string, object>(StringComparer.Ordinal);
            WanVaeConfig.WriteTo(ReadJson(path, "vae"), vae, Krea2Config.Prefix);
            string displayName = Path.GetFileName(Path.TrimEndingDirectorySeparator(Path.GetFullPath(path)));
            return QwenImageSource.Create("diffusers", displayName, path, weights, Krea2Config.Architecture, config.WriteTo, tensors.Resources, vae, textEncoder);
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

    private static JsonElement ReadJson(string root, string component, string file = "config.json")
    {
        using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(root, component, file)));
        return document.RootElement.Clone();
    }
}
