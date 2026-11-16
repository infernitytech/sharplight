using Sharplight.Import.Gguf;
using Sharplight.Import.HuggingFace;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Recognizes single-file checkpoints of diffusion-transformer image models that are not supported yet (FLUX, SD3, ...), so that opening
/// one says so by name instead of failing on the first unexpected tensor: their GGUF files (city96's <c>flux</c> / <c>sd3</c> architectures,
/// stable-diffusion.cpp conversions under <c>model.diffusion_model.</c>) and their safetensors files are claimed by the diffusers importer
/// and rejected with "not supported yet".
/// </summary>
internal static class DiffusionTransformerCheckpoint
{
    private const string Prefix = "model.diffusion_model.";

    /// <summary>The family name of an unsupported diffusion-transformer checkpoint, or null.</summary>
    /// <param name="path">A file.</param>
    public static string? FamilyOf(string path)
    {
        if (!File.Exists(path))
        {
            return null;
        }

        try
        {
            if (CheckpointTensors.IsGguf(path))
            {
                using var gguf = GgufFile.Open(path);
                return FamilyOf(gguf.Architecture, gguf.Tensors.Select(tensor => tensor.Name));
            }

            if (!path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
            {
                return null;
            }

            using var file = SafetensorsFile.Open(path);
            return FamilyOf(null, file.Tensors.Keys);
        }
        catch (Exception error) when (error is IOException or InvalidDataException or NotSupportedException or UnauthorizedAccessException)
        {
            return null;
        }
    }

    /// <summary>The family name from a GGUF architecture id (or null) and the tensor names, or null when it is not a diffusion transformer.</summary>
    /// <param name="ggufArchitecture">The GGUF <c>general.architecture</c>, or null.</param>
    /// <param name="names">The tensor names.</param>
    public static string? FamilyOf(string? ggufArchitecture, IEnumerable<string> names)
    {
        switch (ggufArchitecture?.ToLowerInvariant())
        {
            case "flux":
                return "FLUX (diffusion transformer)";
            case "sd3":
                return "Stable Diffusion 3 (diffusion transformer)";
            case "aura" or "auraflow":
                return "AuraFlow (diffusion transformer)";
            case "hyvid" or "wan" or "ltxv" or "cosmos" or "lumina2" or "hidream" or "qwen_image" or "chroma":
                return $"{ggufArchitecture} (diffusion transformer)";
        }

        var stripped = names.Select(name => name.StartsWith(Prefix, StringComparison.Ordinal) ? name[Prefix.Length..] : name).ToList();
        if (stripped.Any(name => name.StartsWith("double_blocks.", StringComparison.Ordinal) || name.StartsWith("single_blocks.", StringComparison.Ordinal)))
        {
            return stripped.Any(name => name.StartsWith("distilled_guidance_layer.", StringComparison.Ordinal))
                ? "Chroma (diffusion transformer)"
                : "FLUX (diffusion transformer)";
        }

        if (stripped.Any(name => name.StartsWith("joint_blocks.", StringComparison.Ordinal)))
        {
            return "Stable Diffusion 3 (diffusion transformer)";
        }

        return null;
    }
}
