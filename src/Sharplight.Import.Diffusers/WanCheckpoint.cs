using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Reads transformer-only Wan checkpoints in the original Wan tensor names: GGUF files (city96 / QuantStack, any ggml type) and
/// single <c>.safetensors</c> files (also with ComfyUI's <c>model.diffusion_model.</c> prefix). The text encoder, tokenizer and VAE come as
/// auxiliary sources (a Wan diffusers folder of the same family, merged by the <c>WanCompanionAdapter</c>).
/// </summary>
internal static class WanCheckpoint
{
    private static readonly string[] s_prefixes = ["", "model.diffusion_model."];

    /// <summary>The prefix of the original-name Wan transformer tensors, or null when the names are not a Wan transformer's.</summary>
    /// <param name="names">The tensor names of the file.</param>
    public static string? TransformerPrefix(IReadOnlyCollection<string> names)
        => s_prefixes.FirstOrDefault(prefix => names.Contains(prefix + "patch_embedding.weight") && names.Contains(prefix + "blocks.0.self_attn.q.weight")
            && names.Contains(prefix + "head.modulation"));

    /// <summary>Opens the file as a transformer-only canonical <c>wan</c> source (configured from the tensor shapes).</summary>
    /// <param name="path">The checkpoint.</param>
    public static ModelSource Open(string path)
    {
        var tensors = new CheckpointTensors();
        try
        {
            tensors.Add(path);
            string prefix = TransformerPrefix([.. tensors.Tensors.Keys]) ?? throw new NotSupportedException($"'{path}' is not a Wan transformer.");
            var weights = new CanonicalWeights();
            foreach (var (name, weight) in tensors.Tensors)
            {
                if (name.StartsWith(prefix, StringComparison.Ordinal))
                {
                    WanNames.AddTransformer(name[prefix.Length..], weight, weights);
                }
            }

            var metadata = new Dictionary<string, object>(WanConfig.FromWeights(weights.Weights).Metadata, StringComparer.Ordinal)
            {
                [CanonicalKeys.Name] = Path.GetFileNameWithoutExtension(path),
                [CanonicalKeys.Architecture] = WanConfig.Architecture,
            };
            return new ModelSource
            {
                Format = CheckpointTensors.IsGguf(path) ? "gguf" : "safetensors",
                Architecture = WanConfig.Architecture,
                Metadata = metadata,
                Weights = weights.Weights,
                OwnedResources = tensors.Resources,
                Path = path,
            };
        }
        catch
        {
            tensors.DisposeAll();
            throw;
        }
    }
}
