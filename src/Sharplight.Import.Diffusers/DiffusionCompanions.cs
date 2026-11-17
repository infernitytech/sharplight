using Sharplight.Import.Gguf;
using Sharplight.Models;
using Sharplight.Vision;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Finds the companion files of a transformer-only diffusion checkpoint (FLUX: CLIP-L, the T5 encoder and the 16-channel VAE; SD3: CLIP-L,
/// CLIP-G, the T5 encoder and the 16-channel VAE; the T5 encoder as GGUF or safetensors; Z-Image / Qwen-Image: their language model
/// and VAE), looked for beside the file and in the sibling folders of the usual model directory layouts (ComfyUI's
/// <c>clip</c>, <c>text_encoders</c> and <c>vae</c> next to <c>unet</c> / <c>diffusion_models</c>). Candidates are recognized by their
/// tensors, not only their names; of several, GGUF T5 files come first (they carry their tokenizer), then by file name.
/// </summary>
internal static class DiffusionCompanions
{
    private static readonly string[] s_siblingFolders = ["clip", "text_encoders", "text_encoder", "t5", "vae"];

    /// <summary>The companion files by auxiliary role (roles without a candidate are left out).</summary>
    /// <param name="transformerPath">The transformer file.</param>
    /// <param name="family">The model family (which encoders it needs, and in which roles).</param>
    public static IReadOnlyDictionary<string, string> Find(string transformerPath, CompanionFamily family)
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
            string? role = RoleOf(file, family);
            if (role is not null && !found.ContainsKey(role))
            {
                found[role] = file;
            }
        }

        // Diffusers pipeline folders beside the transformer's folder (e.g. the official repository downloaded next to a GGUF folder):
        // their VAE file and, for Z-Image, their text encoder folder.
        var pipelines = Path.GetDirectoryName(folder) is { } root
            ? Directory.EnumerateDirectories(root).Where(d => File.Exists(Path.Combine(d, "model_index.json"))).Order(StringComparer.Ordinal).ToList()
            : [];
        foreach (string pipeline in pipelines)
        {
            string vae = Path.Combine(pipeline, "vae", "diffusion_pytorch_model.safetensors");
            if (!found.ContainsKey(AuxiliaryRoles.Vae) && File.Exists(vae) && IsFamilyVae(vae, family))
            {
                found[AuxiliaryRoles.Vae] = vae;
            }

            string textEncoder = Path.Combine(pipeline, "text_encoder");
            string? languageModel = family switch
            {
                CompanionFamily.ZImage => "qwen3",
                CompanionFamily.QwenImage => "qwen2_5_vl",
                CompanionFamily.QwenImage21 => "qwen3_vl",
                CompanionFamily.Flux2 => "qwen3",
                CompanionFamily.Flux2Dev => "mistral3",
                CompanionFamily.Krea2 => "qwen3_vl",
                _ => null,
            };
            if (languageModel is not null && !found.ContainsKey(AuxiliaryRoles.TextEncoder) && ModelType(textEncoder) == languageModel)
            {
                found[AuxiliaryRoles.TextEncoder] = textEncoder;
            }
        }

        return found;
    }

    /// <summary>The <c>model_type</c> of a Hugging Face model folder (Z-Image's text encoder: <c>qwen3</c>; Qwen-Image's: <c>qwen2_5_vl</c>), or null.</summary>
    private static string? ModelType(string folder)
    {
        string config = Path.Combine(folder, "config.json");
        if (!File.Exists(config))
        {
            return null;
        }

        using var document = System.Text.Json.JsonDocument.Parse(File.ReadAllBytes(config));
        return document.RootElement.TryGetProperty("model_type", out var type) ? type.GetString() : null;
    }

    /// <summary>Whether a VAE file is the family's: the causal 3-D VAE for Qwen-Image, a 16-channel KL autoencoder for the others.</summary>
    private static bool IsFamilyVae(string file, CompanionFamily family)
        => family is CompanionFamily.QwenImage or CompanionFamily.QwenImage21 or CompanionFamily.Krea2
            ? ComponentCheckpoint.ClassifyFile(file) == ComponentCheckpoint.WanAutoencoderArchitecture
            : family is CompanionFamily.Flux2 or CompanionFamily.Flux2Dev
                ? ComponentCheckpoint.ClassifyFile(file) == ComponentCheckpoint.AutoencoderArchitecture && HasBatchNorm(file)
                : ComponentCheckpoint.ClassifyFile(file) == ComponentCheckpoint.AutoencoderArchitecture && IsSixteenChannelVae(file);

    /// <summary>Whether a VAE file carries FLUX.2's batch-norm statistics.</summary>
    private static bool HasBatchNorm(string file)
    {
        using var safetensors = Sharplight.Import.HuggingFace.SafetensorsFile.Open(file);
        return safetensors.Tensors.ContainsKey("bn.running_mean");
    }

    /// <summary>The companion files of a family, in presentation order.</summary>
    /// <param name="family">The model family.</param>
    public static IReadOnlyList<CompanionFile> Describe(CompanionFamily family)
    {
        var clipL = new CompanionFile(AuxiliaryRoles.TextEncoder, "Text encoder (CLIP-L)", "clip_l.safetensors");
        var vae = new CompanionFile(AuxiliaryRoles.Vae, "VAE", family == CompanionFamily.Sd3 ? "the SD3 VAE (diffusion_pytorch_model.safetensors)" : "ae.safetensors");
        const string t5 = "t5-v1_1-xxl-encoder-Q8_0.gguf or t5xxl_fp16.safetensors";
        if (family == CompanionFamily.ZImage)
        {
            return [new CompanionFile(AuxiliaryRoles.TextEncoder, "Text encoder (Qwen3-4B)", "Qwen3-4B-Q8_0.gguf"), vae];
        }

        if (family == CompanionFamily.Flux2Dev)
        {
            return
            [
                new CompanionFile(AuxiliaryRoles.TextEncoder, "Text encoder (Mistral Small 3.2 24B)", "Mistral-Small-3.2-24B-Instruct-2506-Q8_0.gguf or the pipeline's text_encoder folder"),
                new CompanionFile(AuxiliaryRoles.Vae, "VAE", "flux2-vae.safetensors or the pipeline's vae file"),
            ];
        }

        if (family == CompanionFamily.Flux2)
        {
            return
            [
                new CompanionFile(AuxiliaryRoles.TextEncoder, "Text encoder (Qwen3)", "Qwen3-4B-Q8_0.gguf (Klein 4B) or the pipeline's text_encoder folder"),
                new CompanionFile(AuxiliaryRoles.Vae, "VAE", "flux2-vae.safetensors or the pipeline's vae file"),
            ];
        }

        if (family == CompanionFamily.Krea2)
        {
            return
            [
                new CompanionFile(AuxiliaryRoles.TextEncoder, "Text encoder (Qwen3-VL-4B)", "Qwen3-VL-4B-Instruct-Q8_0.gguf or the pipeline's text_encoder folder"),
                new CompanionFile(AuxiliaryRoles.Vae, "VAE", "qwen_image_vae.safetensors or the pipeline's vae file"),
            ];
        }

        if (family == CompanionFamily.QwenImage21)
        {
            return
            [
                new CompanionFile(AuxiliaryRoles.TextEncoder, "Text encoder (Qwen3-VL-8B)", "Qwen3-VL-8B-Instruct-Q8_0.gguf or the pipeline's text_encoder folder"),
                new CompanionFile(AuxiliaryRoles.Vae, "VAE", "the pipeline's vae/diffusion_pytorch_model.safetensors"),
            ];
        }

        if (family == CompanionFamily.QwenImage)
        {
            return
            [
                new CompanionFile(AuxiliaryRoles.TextEncoder, "Text encoder (Qwen2.5-VL-7B)", "Qwen2.5-VL-7B-Instruct-Q8_0.gguf or the pipeline's text_encoder folder"),
                new CompanionFile(AuxiliaryRoles.Vae, "VAE", "qwen_image_vae.safetensors"),
            ];
        }

        if (family == CompanionFamily.Chroma)
        {
            return [new CompanionFile(AuxiliaryRoles.TextEncoder, "Text encoder (T5-XXL)", t5), vae];
        }

        return family == CompanionFamily.Sd3
            ?
            [
                clipL,
                new CompanionFile(AuxiliaryRoles.SecondTextEncoder, "Text encoder 2 (CLIP-G)", "clip_g.safetensors"),
                new CompanionFile(AuxiliaryRoles.ThirdTextEncoder, "Text encoder 3 (T5-XXL)", t5),
                vae,
            ]
            : [clipL, new CompanionFile(AuxiliaryRoles.SecondTextEncoder, "Text encoder 2 (T5-XXL)", t5), vae];
    }

    /// <summary>The auxiliary role a file can fill, or null.</summary>
    private static string? RoleOf(string file, CompanionFamily family)
    {
        string t5Role = family switch
        {
            CompanionFamily.Sd3 => AuxiliaryRoles.ThirdTextEncoder,
            CompanionFamily.Chroma => AuxiliaryRoles.TextEncoder,
            _ => AuxiliaryRoles.SecondTextEncoder,
        };
        if (file.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
        {
            return family switch
            {
                CompanionFamily.ZImage => GgufArchitecture(file) == "qwen3" ? AuxiliaryRoles.TextEncoder : null,
                CompanionFamily.QwenImage => GgufArchitecture(file) == "qwen2vl" ? AuxiliaryRoles.TextEncoder : null,
                CompanionFamily.QwenImage21 => GgufArchitecture(file) == "qwen3vl" ? AuxiliaryRoles.TextEncoder : null,
                CompanionFamily.Flux2 => GgufArchitecture(file) == "qwen3" ? AuxiliaryRoles.TextEncoder : null,
                CompanionFamily.Flux2Dev => GgufArchitecture(file) is "llama" or "mistral3" && GgufLayerWidth(file) == 5120 ? AuxiliaryRoles.TextEncoder : null,
                CompanionFamily.Krea2 => GgufArchitecture(file) == "qwen3vl" ? AuxiliaryRoles.TextEncoder : null,
                _ => GgufArchitecture(file) is "t5encoder" or "t5" ? t5Role : null,
            };
        }

        if (family is CompanionFamily.ZImage or CompanionFamily.QwenImage or CompanionFamily.QwenImage21 or CompanionFamily.Flux2 or CompanionFamily.Flux2Dev or CompanionFamily.Krea2)
        {
            return IsFamilyVae(file, family) ? AuxiliaryRoles.Vae : null;
        }

        return ComponentCheckpoint.ClassifyFile(file) switch
        {
            ComponentCheckpoint.ClipArchitecture when family == CompanionFamily.Chroma => null,
            // SD3 tells its two CLIP encoders apart by width (OpenCLIP bigG: 1280); FLUX has only CLIP-L.
            ComponentCheckpoint.ClipArchitecture => family == CompanionFamily.Sd3 && ClipWidth(file) == 1280
                ? AuxiliaryRoles.SecondTextEncoder
                : AuxiliaryRoles.TextEncoder,
            ComponentCheckpoint.T5Architecture => t5Role,
            ComponentCheckpoint.AutoencoderArchitecture when IsSixteenChannelVae(file) => AuxiliaryRoles.Vae,
            _ => null,
        };
    }

    /// <summary>The hidden width of a CLIP text encoder file (768: CLIP-L, 1280: OpenCLIP bigG).</summary>
    private static long ClipWidth(string file)
    {
        using var safetensors = Sharplight.Import.HuggingFace.SafetensorsFile.Open(file);
        return safetensors.Tensors.FirstOrDefault(t => t.Key.EndsWith("embeddings.token_embedding.weight", StringComparison.Ordinal)).Value?.Shape[1] ?? 0;
    }

    /// <summary>The <c>general.architecture</c> of a GGUF file, or null.</summary>
    private static string? GgufArchitecture(string file)
    {
        try
        {
            using var gguf = GgufFile.Open(file);
            return gguf.Metadata.TryGetValue("general.architecture", out var architecture) ? architecture as string : null;
        }
        catch (IOException)
        {
            return null;
        }
        catch (InvalidDataException)
        {
            return null;
        }
    }

    /// <summary>The <c>{architecture}.embedding_length</c> of a GGUF language model (Mistral Small 3.2 24B: 5120), or 0.</summary>
    private static long GgufLayerWidth(string file)
    {
        try
        {
            using var gguf = GgufFile.Open(file);
            return gguf.Metadata.TryGetValue("general.architecture", out var architecture) && architecture is string name
                && gguf.Metadata.TryGetValue(name + ".embedding_length", out var width)
                ? Convert.ToInt64(width, System.Globalization.CultureInfo.InvariantCulture)
                : 0;
        }
        catch (IOException)
        {
            return 0;
        }
        catch (InvalidDataException)
        {
            return 0;
        }
    }

    private static bool IsSixteenChannelVae(string file)
    {
        using var safetensors = Sharplight.Import.HuggingFace.SafetensorsFile.Open(file);
        return safetensors.Tensors.TryGetValue("decoder.conv_in.weight", out var conv) && conv.Shape[1] == 16;
    }
}
