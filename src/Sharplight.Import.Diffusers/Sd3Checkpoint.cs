using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// An SD3 / SD3.5 transformer in one file under the original MMDiT names: Stability's single-file <c>.safetensors</c> releases
/// (<c>model.diffusion_model.*</c>, with the VAE under <c>first_stage_model.*</c>) and the transformer-only GGUF files of city96 /
/// stable-diffusion.cpp (matrices quantized, no prefix). The text encoders (and the VAE, when the file has none) come as auxiliary
/// sources.
/// </summary>
internal static class Sd3Checkpoint
{
    private const string VaePrefix = "first_stage_model.";
    private static readonly string[] s_prefixes = ["", "model.diffusion_model."];

    /// <summary>The prefix of the transformer names when the tensor names are an original-layout SD3 transformer, else null.</summary>
    /// <param name="names">The tensor names.</param>
    public static string? TransformerPrefix(IReadOnlyCollection<string> names)
        => s_prefixes.FirstOrDefault(prefix => Sd3Names.IsOriginal(names, prefix));

    /// <summary>Opens the file as a canonical source (the transformer, and the VAE when the file has it).</summary>
    /// <param name="path">The safetensors or GGUF file.</param>
    public static ModelSource Open(string path)
    {
        var tensors = new CheckpointTensors();
        try
        {
            tensors.Add(path);
            var names = tensors.Tensors.Keys.ToList();
            string prefix = TransformerPrefix(names) ?? throw new NotSupportedException($"'{path}' is not an SD3 transformer.");
            var weights = new CanonicalWeights();
            Sd3Names.AddOriginal(tensors.Tensors, prefix, weights);
            var vaeNames = names.Where(n => n.StartsWith(VaePrefix, StringComparison.Ordinal)).Select(n => n[VaePrefix.Length..]).ToList();
            VaeConfig? vae = null;
            if (vaeNames.Count > 0)
            {
                foreach (var (ldm, diffusers) in LdmNames.Vae(vaeNames))
                {
                    DiffusersNames.AddVae(diffusers, tensors.Tensors[VaePrefix + ldm], weights);
                }

                vae = VaeConfig.FromWeights(weights.Weights, Sd3Config.VaeScalingFactor, []) with { ShiftFactor = Sd3Config.VaeShiftFactor };
            }

            var config = Sd3Config.FromWeights(weights.Weights);
            string format = CheckpointTensors.IsGguf(path) ? "gguf" : "safetensors";
            return Sd3Source.Create(format, Path.GetFileNameWithoutExtension(path), path, weights, config, tensors.Resources, vae);
        }
        catch
        {
            tensors.DisposeAll();
            throw;
        }
    }
}
