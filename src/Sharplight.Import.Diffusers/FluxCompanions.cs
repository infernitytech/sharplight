using Sharplight.Import.Gguf;
using Sharplight.Vision;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Finds the companion files of a transformer-only FLUX checkpoint: the CLIP-L text encoder, the T5 encoder (GGUF or safetensors) and
/// the 16-channel VAE, looked for beside the file and in the sibling folders of the usual model directory layouts (ComfyUI's
/// <c>clip</c>, <c>text_encoders</c> and <c>vae</c> next to <c>unet</c> / <c>diffusion_models</c>). Candidates are recognized by their
/// tensors, not only their names; of several, GGUF T5 files come first (they carry their tokenizer), then by file name.
/// </summary>
internal static class FluxCompanions
{
    private static readonly string[] s_siblingFolders = ["clip", "text_encoders", "text_encoder", "t5", "vae"];

    /// <summary>The companion files by auxiliary role (roles without a candidate are left out).</summary>
    /// <param name="transformerPath">The transformer file.</param>
    public static IReadOnlyDictionary<string, string> Find(string transformerPath)
    {
        var found = new Dictionary<string, string>(StringComparer.Ordinal);
        string? folder = Path.GetDirectoryName(Path.GetFullPath(transformerPath));
        if (folder is null)
        {
            return found;
        }

        var folders = new List<string> { folder };
        if (Path.GetDirectoryName(folder) is { } parent)
        {
            folders.AddRange(s_siblingFolders.Select(name => Path.Combine(parent, name)).Where(Directory.Exists));
        }

        var candidates = folders.SelectMany(f => Directory.EnumerateFiles(f))
            .Where(f => !string.Equals(Path.GetFullPath(f), Path.GetFullPath(transformerPath), StringComparison.Ordinal))
            .Where(f => f.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase) || f.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
            .OrderBy(f => f.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase) ? 0 : 1)
            .ThenBy(f => Path.GetFileName(f), StringComparer.OrdinalIgnoreCase)
            .ToList();
        foreach (string file in candidates)
        {
            string? role = RoleOf(file);
            if (role is not null && !found.ContainsKey(role))
            {
                found[role] = file;
            }
        }

        return found;
    }

    /// <summary>The auxiliary role a file can fill, or null.</summary>
    private static string? RoleOf(string file)
    {
        if (file.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
        {
            return IsT5Gguf(file) ? AuxiliaryRoles.SecondTextEncoder : null;
        }

        return ComponentCheckpoint.ClassifyFile(file) switch
        {
            ComponentCheckpoint.ClipArchitecture => AuxiliaryRoles.TextEncoder,
            ComponentCheckpoint.T5Architecture => AuxiliaryRoles.SecondTextEncoder,
            ComponentCheckpoint.AutoencoderArchitecture when IsSixteenChannelVae(file) => AuxiliaryRoles.Vae,
            _ => null,
        };
    }

    private static bool IsT5Gguf(string file)
    {
        try
        {
            using var gguf = GgufFile.Open(file);
            return gguf.Metadata.TryGetValue("general.architecture", out var architecture) && architecture is "t5encoder" or "t5";
        }
        catch (IOException)
        {
            return false;
        }
        catch (InvalidDataException)
        {
            return false;
        }
    }

    private static bool IsSixteenChannelVae(string file)
    {
        using var safetensors = Sharplight.Import.HuggingFace.SafetensorsFile.Open(file);
        return safetensors.Tensors.TryGetValue("decoder.conv_in.weight", out var conv) && conv.Shape[1] == 16;
    }
}
