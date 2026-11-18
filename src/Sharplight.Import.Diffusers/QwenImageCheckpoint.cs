using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// A Qwen-Image transformer in one file: the transformer-only GGUF files (city96, QuantStack, unsloth: <c>qwen_image</c> architecture id,
/// diffusers names, matrices quantized) and single-file safetensors (ComfyUI's, with or without the <c>model.diffusion_model.</c>
/// prefix). The text encoder and the VAE come as auxiliary sources.
/// </summary>
internal static class QwenImageCheckpoint
{
    private static readonly string[] s_prefixes = ["", "model.diffusion_model."];

    /// <summary>The prefix of the transformer names when the tensor names are a Qwen-Image transformer, else null.</summary>
    /// <param name="names">The tensor names.</param>
    public static string? TransformerPrefix(IReadOnlyCollection<string> names)
        => s_prefixes.FirstOrDefault(prefix => QwenImageNames.IsTransformer(names, prefix));

    /// <summary>Opens the file as a (transformer-only) canonical source.</summary>
    /// <param name="path">The safetensors or GGUF file.</param>
    public static ModelSource Open(string path)
    {
        var tensors = new CheckpointTensors();
        try
        {
            tensors.Add(path);
            string prefix = TransformerPrefix([.. tensors.Tensors.Keys]) ?? throw new NotSupportedException($"'{path}' is not a Qwen-Image transformer.");
            var weights = new CanonicalWeights();
            QwenImageNames.Add(tensors.Tensors, prefix, weights);
            string format = CheckpointTensors.IsGguf(path) ? "gguf" : "safetensors";
            var config = QwenImageConfig.FromWeights(weights.Weights)
                .WithEdit(QwenImageEditVariant.FromFile(path, tensors.Tensors.ContainsKey(prefix + QwenImageNames.ZeroConditionMarker)));
            return QwenImageSource.Create(format, Path.GetFileNameWithoutExtension(path), path, weights, QwenImageConfig.Architecture,
                config.WriteTo, tensors.Resources);
        }
        catch
        {
            tensors.DisposeAll();
            throw;
        }
    }
}
