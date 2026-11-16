using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Importer for diffusion image models: Stable Diffusion 1.x / 2.x / XL (diffusers pipeline folders, original-layout single-file
/// <c>.safetensors</c> checkpoints, PyTorch <c>.ckpt</c> checkpoints read with the restricted unpickler, stable-diffusion.cpp GGUF files)
/// translated into the canonical <c>latent_diffusion</c> source, and FLUX.1 (diffusers folders, original-name single files and
/// transformer-only GGUF files) translated into the canonical <c>flux</c> source; plus the stand-alone component files that come with
/// transformer-only checkpoints (CLIP and T5 text encoders, VAEs), which open as component sources and are found beside the
/// transformer (<see cref="FindCompanions"/>). The <c>LatentDiffusionAdapter</c> and <c>FluxAdapter</c> of Sharplight.Architectures
/// build the image-generation models.
/// <para>
/// Single-file and GGUF checkpoints share their containers with LLM checkpoints; this importer claims them by content (a
/// <c>model.diffusion_model.</c> UNet) and ranks above the generic importers (<see cref="Priority"/>).
/// </para>
/// </summary>
public sealed class DiffusersImporter : IModelImporter
{
    /// <summary>The format id.</summary>
    public const string FormatName = "diffusers";

    /// <inheritdoc/>
    public string Format => FormatName;

    /// <inheritdoc/>
    public int Priority => 10;


    /// <inheritdoc/>
    /// <remarks>
    /// Also claims single-file checkpoints of the diffusion transformers that are not supported yet (SD3, Chroma, ...), which
    /// <see cref="Open"/> rejects by name, instead of leaving them to an importer that would fail on an unknown architecture.
    /// </remarks>
    public bool CanOpen(string path)
        => DiffusersFolder.IsFolder(path) || SingleFileCheckpoint.Kind(path) is not null || DiffusionTransformerCheckpoint.FamilyOf(path) is not null;

    /// <summary>Opens a diffusers folder, a single-file checkpoint, a transformer-only FLUX file or a stand-alone component file.</summary>
    /// <param name="path">The pipeline folder, or a <c>.safetensors</c> / <c>.ckpt</c> / <c>.gguf</c> checkpoint.</param>
    /// <returns>The canonical source; it owns the memory-mapped files.</returns>
    /// <exception cref="NotSupportedException">The model is not a supported diffusion model (e.g. a diffusion transformer other than FLUX.1, not supported yet).</exception>
    public ModelSource Open(string path)
    {
        if (Directory.Exists(path))
        {
            return DiffusersFolder.Open(path);
        }

        string? kind = SingleFileCheckpoint.Kind(path);
        string? family = DiffusionTransformerCheckpoint.FamilyOf(path);
        if (family is not null && !(kind == FluxConfig.Architecture && family.StartsWith("FLUX", StringComparison.Ordinal)))
        {
            throw new NotSupportedException($"Not supported yet: {family}. Sharplight generates images with the Stable Diffusion UNet models "
                + $"(SD 1.x, SD 2.x, SDXL) and FLUX.1; {Path.GetFileName(path)} is another diffusion-transformer model.");
        }

        return kind switch
        {
            FluxConfig.Architecture => FluxCheckpoint.Open(path),
            ComponentCheckpoint.ClipArchitecture or ComponentCheckpoint.AutoencoderArchitecture or ComponentCheckpoint.T5Architecture => ComponentCheckpoint.Open(path),
            _ => SingleFileCheckpoint.Open(path),
        };
    }

    /// <summary>The text encoders and VAE a transformer-only FLUX file needs, found beside it (<see cref="FluxCompanions"/>).</summary>
    /// <param name="path">The model file.</param>
    public IReadOnlyDictionary<string, string> FindCompanions(string path)
        => File.Exists(path) && SingleFileCheckpoint.Kind(path) == FluxConfig.Architecture ? FluxCompanions.Find(path) : new Dictionary<string, string>();
}
