using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// A Qwen-Image 2.1 transformer in one file: the transformer-only GGUF files (unsloth: no metadata, ComfyUI names with the fused SwiGLU,
/// matrices quantized) and ComfyUI single files (with or without the <c>model.diffusion_model.</c>
/// prefix). The text encoder and the VAE come as auxiliary sources.
/// </summary>
internal static class QwenImage21Checkpoint
{
    private static readonly string[] s_prefixes = ["", "model.diffusion_model."];

    /// <summary>The prefix of the transformer names when the tensor names are a Qwen-Image 2.1 transformer, else null.</summary>
    /// <param name="names">The tensor names.</param>
    public static string? TransformerPrefix(IReadOnlyCollection<string> names)
        => s_prefixes.FirstOrDefault(prefix => QwenImage21Names.IsTransformer(names, prefix));

    /// <summary>Opens the file as a (transformer-only) canonical source.</summary>
    /// <param name="path">The safetensors or GGUF file.</param>
    public static ModelSource Open(string path)
    {
        var tensors = new CheckpointTensors();
        try
        {
            tensors.Add(path);
            string prefix = TransformerPrefix([.. tensors.Tensors.Keys]) ?? throw new NotSupportedException($"'{path}' is not a Qwen-Image 2.1 transformer.");
            var weights = new CanonicalWeights();
            QwenImage21Names.Add(tensors.Tensors, prefix, weights);
            string format = CheckpointTensors.IsGguf(path) ? "gguf" : "safetensors";
            var config = QwenImage21Config.FromWeights(weights.Weights);
            if (Path.GetFileName(path).Contains("turbo", StringComparison.OrdinalIgnoreCase))
            {
                // The Turbo transformer is the base's layout: recognized by its published name, it samples its own 8 steps.
                config = config with { SampleSigmas = QwenImage21Config.TurboSigmas };
            }

            return QwenImageSource.Create(format, Path.GetFileNameWithoutExtension(path), path, weights, QwenImage21Config.Architecture, config.WriteTo,
                tensors.Resources);
        }
        catch
        {
            tensors.DisposeAll();
            throw;
        }
    }
}
