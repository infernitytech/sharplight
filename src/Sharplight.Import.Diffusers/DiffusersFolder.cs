using System.Text.Json;
using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Reads a diffusers pipeline folder (<c>model_index.json</c> with <c>unet/</c>, <c>vae/</c>, <c>text_encoder/</c>, <c>tokenizer/</c> and
/// <c>scheduler/</c>) of a Stable Diffusion 1.x / 2.x pipeline.
/// </summary>
internal static class DiffusersFolder
{
    /// <summary>Whether <paramref name="path"/> is a diffusers pipeline folder with a UNet.</summary>
    /// <param name="path">A directory.</param>
    public static bool IsFolder(string path)
        => Directory.Exists(path) && File.Exists(Path.Combine(path, "model_index.json")) && Directory.Exists(Path.Combine(path, "unet"))
           && Directory.Exists(Path.Combine(path, "vae"));

    /// <summary>Opens the folder as a canonical source.</summary>
    /// <param name="path">The pipeline folder.</param>
    /// <exception cref="NotSupportedException">The pipeline is not a single-text-encoder UNet pipeline.</exception>
    public static ModelSource Open(string path)
    {
        if (Directory.Exists(Path.Combine(path, "text_encoder_2")) || Directory.Exists(Path.Combine(path, "transformer")))
        {
            throw new NotSupportedException($"'{path}': only Stable Diffusion 1.x / 2.x pipelines (one CLIP text encoder and a UNet) are supported so far.");
        }

        var tensors = new CheckpointTensors();
        try
        {
            AddComponent(tensors, path, "unet");
            AddComponent(tensors, path, "vae");
            AddComponent(tensors, path, "text_encoder");
            var weights = new CanonicalWeights();
            foreach (var (name, weight) in tensors.Tensors)
            {
                int dot = name.IndexOf('.');
                string component = name[..dot];
                string rest = name[(dot + 1)..];
                switch (component)
                {
                    case "unet":
                        DiffusersNames.AddUNet(rest, weight, weights);
                        break;
                    case "vae":
                        DiffusersNames.AddVae(rest, weight, weights);
                        break;
                    default:
                        DiffusersNames.AddClipText(rest, weight, LatentDiffusionSource.TextEncoderPrefix, weights);
                        break;
                }
            }

            var unet = UNetConfig.FromDiffusers(ReadJson(path, "unet", "config.json"));
            var vae = VaeConfig.FromDiffusers(ReadJson(path, "vae", "config.json"), LatentDiffusionSource.StableDiffusion1RgbFactors);
            var text = ClipTextConfig.FromTransformers(ReadJson(path, "text_encoder", "config.json"));
            var vocabulary = ClipVocabulary.FromFolder(Path.Combine(path, "tokenizer"));
            string schedulerPath = Path.Combine(path, "scheduler", "scheduler_config.json");
            var scheduler = File.Exists(schedulerPath) ? SchedulerSettings.FromDiffusers(File.ReadAllText(schedulerPath)) : new SchedulerSettings();
            string displayName = Path.GetFileName(Path.TrimEndingDirectorySeparator(Path.GetFullPath(path)));
            return LatentDiffusionSource.Create("diffusers", displayName, path, weights, unet, vae, text, vocabulary, scheduler, tensors.Resources);
        }
        catch
        {
            tensors.DisposeAll();
            throw;
        }
    }

    /// <summary>Adds the safetensors files of one component folder, preferring full-precision files over <c>.fp16</c> variants.</summary>
    private static void AddComponent(CheckpointTensors tensors, string root, string component)
    {
        string folder = Path.Combine(root, component);
        var files = Directory.GetFiles(folder, "*.safetensors").Order(StringComparer.Ordinal).ToList();
        var plain = files.Where(f => !Path.GetFileName(f).Contains(".fp16.", StringComparison.Ordinal)).ToList();
        var chosen = plain.Count > 0 ? plain : files;
        if (chosen.Count == 0)
        {
            throw new FileNotFoundException($"'{folder}' has no .safetensors weights.");
        }

        foreach (string file in chosen)
        {
            tensors.Add(file, component + ".");
        }
    }

    private static JsonElement ReadJson(string root, string component, string file)
    {
        using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(root, component, file)));
        return document.RootElement.Clone();
    }
}
