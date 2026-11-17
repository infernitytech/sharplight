using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// A FLUX.2 transformer in one file: the original-name single files (Black Forest Labs', ComfyUI's) and the transformer-only GGUF files
/// (unsloth, leejet: the <c>flux</c> architecture id of FLUX.1, told apart by the shared modulation tensors). The Qwen3 text encoder and the
/// VAE come as auxiliary sources. File names with <c>base</c> are the undistilled base models (guidance, 50 steps).
/// </summary>
internal static class Flux2Checkpoint
{
    /// <summary>The prefix of the transformer names when the tensor names are a FLUX.2 transformer, else null.</summary>
    /// <param name="names">The tensor names.</param>
    public static string? TransformerPrefix(IReadOnlyCollection<string> names) => Flux2Names.OriginalPrefix(names);

    /// <summary>Opens the file as a (transformer-only) canonical source.</summary>
    /// <param name="path">The safetensors or GGUF file.</param>
    public static ModelSource Open(string path)
    {
        var tensors = new CheckpointTensors();
        try
        {
            tensors.Add(path);
            string prefix = TransformerPrefix([.. tensors.Tensors.Keys]) ?? throw new NotSupportedException($"'{path}' is not a FLUX.2 transformer.");
            var weights = new CanonicalWeights();
            Flux2Names.AddOriginal(tensors.Tensors, prefix, weights);
            var config = Flux2Config.FromWeights(weights.Weights) with
            {
                Distilled = !Path.GetFileName(path).Contains("base", StringComparison.OrdinalIgnoreCase),
            };
            string format = CheckpointTensors.IsGguf(path) ? "gguf" : "safetensors";
            return QwenImageSource.Create(format, Path.GetFileNameWithoutExtension(path), path, weights, Flux2Config.Architecture, config.WriteTo, tensors.Resources);
        }
        catch
        {
            tensors.DisposeAll();
            throw;
        }
    }
}
