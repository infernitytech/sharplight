using System.Text.Json;
using Sharplight.Import.HuggingFace;
using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Reads a diffusers Qwen-Image pipeline folder (<c>transformer/</c> with a <c>QwenImageTransformer2DModel</c>, <c>vae/</c> with an
/// <c>AutoencoderKLQwenImage</c>, <c>text_encoder/</c> (Qwen2.5-VL, opened by the Hugging Face importer; its vision tower is unused),
/// <c>tokenizer/</c>, <c>scheduler/</c>).
/// </summary>
internal static class QwenImageFolder
{
    /// <summary>Whether the folder holds a Qwen-Image transformer.</summary>
    /// <param name="path">A directory.</param>
    public static bool IsFolder(string path)
    {
        string config = Path.Combine(path, "transformer", "config.json");
        if (!File.Exists(config))
        {
            return false;
        }

        using var document = JsonDocument.Parse(File.ReadAllBytes(config));
        return document.RootElement.String("_class_name", "") == "QwenImageTransformer2DModel";
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

            QwenImageNames.Add(transformer, "", weights);
            textEncoder = HuggingFaceImporter.Open(Path.Combine(path, "text_encoder"));
            string schedulerPath = Path.Combine(path, "scheduler", "scheduler_config.json");
            var config = QwenImageConfig.FromWeights(weights.Weights)
                .WithDiffusers(File.Exists(schedulerPath) ? File.ReadAllText(schedulerPath) : null, ReadJson(path, "transformer"));
            var vae = new Dictionary<string, object>(StringComparer.Ordinal);
            WanVaeConfig.WriteTo(ReadJson(path, "vae"), vae, QwenImageConfig.Prefix);
            string displayName = Path.GetFileName(Path.TrimEndingDirectorySeparator(Path.GetFullPath(path)));
            return QwenImageSource.Create("diffusers", displayName, path, weights, config, tensors.Resources, vae, textEncoder);
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
