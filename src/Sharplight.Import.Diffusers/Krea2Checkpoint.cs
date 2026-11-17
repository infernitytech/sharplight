using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// A Krea 2 transformer in one file, in the original Krea names: the release's <c>turbo.safetensors</c> / <c>raw.safetensors</c>, ComfyUI's
/// files (with or without <c>model.diffusion_model.</c>) and the transformer-only GGUF files (architecture <c>krea2</c>, some conversions
/// mislabelled <c>qwen_image</c>). The Qwen3-VL text encoder and the Qwen-Image VAE come as auxiliary sources. File names with <c>raw</c> or
/// <c>base</c> are the undistilled base model (guidance, 28 steps); others are taken for the distilled Turbo and its fine-tunes.
/// </summary>
internal static class Krea2Checkpoint
{
    /// <summary>The prefix of the transformer names when the tensor names are a Krea 2 transformer, else null.</summary>
    /// <param name="names">The tensor names.</param>
    public static string? TransformerPrefix(IReadOnlyCollection<string> names) => Krea2Names.OriginalPrefix(names);

    /// <summary>Opens the file as a (transformer-only) canonical source.</summary>
    /// <param name="path">The safetensors or GGUF file.</param>
    public static ModelSource Open(string path)
    {
        var tensors = new CheckpointTensors();
        try
        {
            tensors.Add(path);
            string prefix = TransformerPrefix([.. tensors.Tensors.Keys]) ?? throw new NotSupportedException($"'{path}' is not a Krea 2 transformer.");
            var weights = new CanonicalWeights();
            Krea2Names.Add(tensors.Tensors, prefix, weights);
            string file = Path.GetFileName(path);
            var config = Krea2Config.FromWeights(weights.Weights) with
            {
                Distilled = !file.Contains("raw", StringComparison.OrdinalIgnoreCase) && !file.Contains("base", StringComparison.OrdinalIgnoreCase),
            };
            string format = CheckpointTensors.IsGguf(path) ? "gguf" : "safetensors";
            return QwenImageSource.Create(format, Path.GetFileNameWithoutExtension(path), path, weights, Krea2Config.Architecture, config.WriteTo, tensors.Resources);
        }
        catch
        {
            tensors.DisposeAll();
            throw;
        }
    }
}
