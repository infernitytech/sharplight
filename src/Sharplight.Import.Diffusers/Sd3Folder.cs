using System.Text.Json;
using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Reads a diffusers SD3 / SD3.5 pipeline folder (<c>transformer/</c> with an <c>SD3Transformer2DModel</c>, <c>vae/</c>,
/// <c>text_encoder/</c> (CLIP-L), <c>text_encoder_2/</c> (CLIP-G), <c>text_encoder_3/</c> (T5), <c>tokenizer/</c>, <c>tokenizer_2/</c>,
/// <c>tokenizer_3/</c>, <c>scheduler/</c>).
/// </summary>
internal static class Sd3Folder
{
    /// <summary>Whether the folder holds an SD3 transformer.</summary>
    /// <param name="path">A directory.</param>
    public static bool IsFolder(string path)
    {
        string config = Path.Combine(path, "transformer", "config.json");
        if (!File.Exists(config))
        {
            return false;
        }

        using var document = JsonDocument.Parse(File.ReadAllBytes(config));
        return document.RootElement.String("_class_name", "") == "SD3Transformer2DModel";
    }

    /// <summary>Opens the folder as a canonical source.</summary>
    /// <param name="path">The pipeline folder.</param>
    /// <exception cref="NotSupportedException">The folder lacks the T5 encoder.</exception>
    public static ModelSource Open(string path)
    {
        if (!Directory.Exists(Path.Combine(path, "text_encoder_3")))
        {
            throw new NotSupportedException($"'{path}': SD3 pipelines without the T5 encoder (text_encoder_3) are not supported yet.");
        }

        var tensors = new CheckpointTensors();
        var resources = new List<IDisposable>();
        try
        {
            DiffusersFolder.AddComponent(tensors, path, "transformer");
            DiffusersFolder.AddComponent(tensors, path, "vae");
            DiffusersFolder.AddComponent(tensors, path, "text_encoder");
            DiffusersFolder.AddComponent(tensors, path, "text_encoder_2");
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
                    case "text_encoder_2":
                        DiffusersNames.AddClipText(rest, weight, Sd3Source.SecondTextEncoderPrefix, weights);
                        break;
                    default:
                        DiffusersNames.AddClipText(rest, weight, LatentDiffusionSource.TextEncoderPrefix, weights);
                        break;
                }
            }

            Sd3Names.AddDiffusers(transformer, weights);
            resources.AddRange(tensors.Resources);
            var t5 = FluxSource.AddT5Folder(Path.Combine(path, "text_encoder_3"), Path.Combine(path, "tokenizer_3"), weights, resources, Sd3Source.T5Prefix);
            string schedulerPath = Path.Combine(path, "scheduler", "scheduler_config.json");
            var config = Sd3Config.FromWeights(weights.Weights)
                .WithDiffusers(File.Exists(schedulerPath) ? File.ReadAllText(schedulerPath) : null, ReadJson(path, "transformer"));
            var vae = VaeConfig.FromDiffusers(ReadJson(path, "vae"), []);
            var clip = ClipTextConfig.FromTransformers(ReadJson(path, "text_encoder"));
            var clip2 = ClipTextConfig.FromTransformers(ReadJson(path, "text_encoder_2"));
            var vocabulary = ClipVocabulary.FromFolder(Path.Combine(path, "tokenizer"));
            string displayName = Path.GetFileName(Path.TrimEndingDirectorySeparator(Path.GetFullPath(path)));
            string secondTokenizer = Path.Combine(path, "tokenizer_2");
            int? secondPadId = Directory.Exists(secondTokenizer) ? ClipVocabulary.FromFolder(secondTokenizer).PadId : null;
            return Sd3Source.Create("diffusers", displayName, path, weights, config, resources, vae, clip, clip2, vocabulary, t5, secondPadId);
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
