using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Importer for latent-diffusion image models (Stable Diffusion 1.x / 2.x): diffusers pipeline folders, original-layout single-file
/// <c>.safetensors</c> checkpoints, PyTorch <c>.ckpt</c> checkpoints (read with the restricted unpickler) and stable-diffusion.cpp GGUF files, translated into the canonical <c>latent_diffusion</c> model
/// source (canonical UNet / VAE / text encoder weights, their configurations, the CLIP tokenizer and the noise schedule). The
/// <c>LatentDiffusionAdapter</c> of Sharplight.Architectures builds the image-generation model from it.
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
    public bool CanOpen(string path) => DiffusersFolder.IsFolder(path) || SingleFileCheckpoint.IsCheckpoint(path);

    /// <summary>Opens a diffusers folder or a single-file checkpoint.</summary>
    /// <param name="path">The pipeline folder, or a <c>.safetensors</c> / <c>.ckpt</c> / <c>.gguf</c> checkpoint.</param>
    /// <returns>The canonical source; it owns the memory-mapped files.</returns>
    /// <exception cref="NotSupportedException">The model is not a supported latent-diffusion model.</exception>
    public ModelSource Open(string path) => Directory.Exists(path) ? DiffusersFolder.Open(path) : SingleFileCheckpoint.Open(path);
}
