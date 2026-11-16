using System.Text.Json;
using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Reads a diffusers pipeline folder (<c>model_index.json</c> with <c>unet/</c>, <c>vae/</c>, <c>text_encoder/</c>, <c>tokenizer/</c> and
/// <c>scheduler/</c>) of a Stable Diffusion 1.x / 2.x / XL pipeline; FLUX and SD3 folders (<c>transformer/</c>) go to <see cref="FluxFolder"/>
/// and <see cref="Sd3Folder"/>.
/// </summary>
internal static class DiffusersFolder
{
    /// <summary>
    /// Whether <paramref name="path"/> is a diffusers pipeline folder with a UNet and a VAE (with or without <c>model_index.json</c>:
    /// partial downloads often lack it).
    /// </summary>
    /// <param name="path">A directory.</param>
    public static bool IsFolder(string path)
        => Directory.Exists(path) && File.Exists(Path.Combine(path, "vae", "config.json"))
           && (File.Exists(Path.Combine(path, "unet", "config.json")) || FluxFolder.IsFolder(path) || Sd3Folder.IsFolder(path) || ChromaFolder.IsFolder(path));

    /// <summary>Opens the folder as a canonical source.</summary>
    /// <param name="path">The pipeline folder.</param>
    /// <exception cref="NotSupportedException">The pipeline is not a single-text-encoder UNet pipeline.</exception>
    public static ModelSource Open(string path)
    {
        if (FluxFolder.IsFolder(path))
        {
            return FluxFolder.Open(path);
        }

        if (Sd3Folder.IsFolder(path))
        {
            return Sd3Folder.Open(path);
        }

        if (ChromaFolder.IsFolder(path))
        {
            return ChromaFolder.Open(path);
        }

        if (Directory.Exists(Path.Combine(path, "transformer")) || !Directory.Exists(Path.Combine(path, "text_encoder")))
        {
            throw new NotSupportedException($"'{path}': only Stable Diffusion 1.x / 2.x / XL / 3, FLUX.1 and Chroma pipelines are supported so far.");
        }

        bool xl = Directory.Exists(Path.Combine(path, "text_encoder_2"));

        var tensors = new CheckpointTensors();
        try
        {
            AddComponent(tensors, path, "unet");
            AddComponent(tensors, path, "vae");
            AddComponent(tensors, path, "text_encoder");
            if (xl)
            {
                AddComponent(tensors, path, "text_encoder_2");
            }

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
                    case "text_encoder_2":
                        DiffusersNames.AddClipText(rest, weight, LatentDiffusionSource.SecondTextEncoderPrefix, weights);
                        break;
                    default:
                        DiffusersNames.AddClipText(rest, weight, LatentDiffusionSource.TextEncoderPrefix, weights);
                        break;
                }
            }

            var unet = UNetConfig.FromDiffusers(ReadJson(path, "unet", "config.json"));
            var vae = VaeConfig.FromDiffusers(ReadJson(path, "vae", "config.json"), LatentDiffusionSource.StableDiffusion1RgbFactors);
            var text = ClipTextConfig.FromTransformers(ReadJson(path, "text_encoder", "config.json"));
            ClipTextConfig? secondText = null;
            int secondPadId = -1;
            if (xl)
            {
                // SDXL conditions on both encoders' penultimate layers, without the final norm.
                text = text with { ClipSkip = 2, FinalNorm = false };
                secondText = ClipTextConfig.FromTransformers(ReadJson(path, "text_encoder_2", "config.json")) with { ClipSkip = 2, FinalNorm = false };
                string secondTokenizer = Path.Combine(path, "tokenizer_2");
                secondPadId = (Directory.Exists(secondTokenizer) ? ClipVocabulary.FromFolder(secondTokenizer) : ClipVocabulary.OpenAi("!")).PadId;
            }

            bool zeroEmptyNegative = File.Exists(Path.Combine(path, "model_index.json"))
                && ReadJson(path, "", "model_index.json").Bool("force_zeros_for_empty_prompt", false);
            var vocabulary = ClipVocabulary.FromFolder(Path.Combine(path, "tokenizer"));
            string schedulerPath = Path.Combine(path, "scheduler", "scheduler_config.json");
            var scheduler = File.Exists(schedulerPath) ? SchedulerSettings.FromDiffusers(File.ReadAllText(schedulerPath)) : new SchedulerSettings();
            string displayName = Path.GetFileName(Path.TrimEndingDirectorySeparator(Path.GetFullPath(path)));
            return LatentDiffusionSource.Create("diffusers", displayName, path, weights, unet, vae, text, vocabulary, scheduler, tensors.Resources,
                secondText, secondPadId, zeroEmptyNegative);
        }
        catch
        {
            tensors.DisposeAll();
            throw;
        }
    }

    /// <summary>Adds the safetensors files of one component folder, preferring full-precision files over <c>.fp16</c> variants.</summary>
    public static void AddComponent(CheckpointTensors tensors, string root, string component)
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
