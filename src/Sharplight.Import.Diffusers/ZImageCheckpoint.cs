using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// A Z-Image transformer in one file: the transformer-only GGUF files (unsloth, stable-diffusion.cpp; matrices quantized) and single-file
/// safetensors (ComfyUI's, with or without the <c>model.diffusion_model.</c> prefix), in the original or the diffusers names. The text
/// encoder and the VAE come as auxiliary sources.
/// </summary>
internal static class ZImageCheckpoint
{
    private static readonly string[] s_prefixes = ["", "model.diffusion_model."];

    /// <summary>The prefix of the transformer names when the tensor names are a Z-Image transformer, else null.</summary>
    /// <param name="names">The tensor names.</param>
    public static string? TransformerPrefix(IReadOnlyCollection<string> names)
        => s_prefixes.FirstOrDefault(prefix => ZImageNames.IsOriginal(names, prefix));

    /// <summary>Opens the file as a (transformer-only) canonical source.</summary>
    /// <param name="path">The safetensors or GGUF file.</param>
    public static ModelSource Open(string path)
    {
        var tensors = new CheckpointTensors();
        try
        {
            tensors.Add(path);
            string prefix = TransformerPrefix([.. tensors.Tensors.Keys]) ?? throw new NotSupportedException($"'{path}' is not a Z-Image transformer.");
            var weights = new CanonicalWeights();
            ZImageNames.Add(tensors.Tensors, prefix, weights);
            string format = CheckpointTensors.IsGguf(path) ? "gguf" : "safetensors";
            return ZImageSource.Create(format, Path.GetFileNameWithoutExtension(path), path, weights, ZImageConfig.FromWeights(weights.Weights).WithFileName(Path.GetFileName(path)), tensors.Resources);
        }
        catch
        {
            tensors.DisposeAll();
            throw;
        }
    }
}
