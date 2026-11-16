using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// A FLUX (or Chroma) transformer in one file under the original (Black Forest Labs) names: the official single-file <c>.safetensors</c>
/// releases and the transformer-only GGUF files of city96 / stable-diffusion.cpp (matrices quantized), with or without the
/// <c>model.diffusion_model.</c> prefix of ComfyUI checkpoints. Such files hold the transformer only; the text encoders and the VAE come
/// as auxiliary sources (<see cref="DiffusionCompanions"/>).
/// </summary>
internal static class FluxCheckpoint
{
    private static readonly string[] s_prefixes = ["", "model.diffusion_model."];

    /// <summary>The prefix of the transformer names when the tensor names are an original-layout FLUX transformer, else null.</summary>
    /// <param name="names">The tensor names.</param>
    public static string? TransformerPrefix(IReadOnlyCollection<string> names)
        => s_prefixes.FirstOrDefault(prefix => FluxNames.IsOriginal(names, prefix));

    /// <summary>Opens the file as a (transformer-only) canonical source.</summary>
    /// <param name="path">The safetensors or GGUF file.</param>
    public static ModelSource Open(string path)
    {
        var tensors = new CheckpointTensors();
        try
        {
            tensors.Add(path);
            string prefix = TransformerPrefix([.. tensors.Tensors.Keys]) ?? throw new NotSupportedException($"'{path}' is not a FLUX transformer.");
            var weights = new CanonicalWeights();
            FluxNames.AddOriginal(tensors.Tensors, prefix, weights);
            string format = CheckpointTensors.IsGguf(path) ? "gguf" : "safetensors";
            if (ChromaConfig.IsChroma(weights.Weights))
            {
                return ChromaSource.Create(format, Path.GetFileNameWithoutExtension(path), path, weights, ChromaConfig.FromWeights(weights.Weights), tensors.Resources);
            }

            var config = FluxConfig.FromWeights(weights.Weights);
            return FluxSource.Create(format, Path.GetFileNameWithoutExtension(path), path, weights, config, tensors.Resources);
        }
        catch
        {
            tensors.DisposeAll();
            throw;
        }
    }
}
