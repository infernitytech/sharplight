using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Importer for diffusion image models: Stable Diffusion 1.x / 2.x / XL (diffusers pipeline folders, original-layout single-file
/// <c>.safetensors</c> checkpoints, PyTorch <c>.ckpt</c> checkpoints read with the restricted unpickler, stable-diffusion.cpp GGUF files)
/// translated into the canonical <c>latent_diffusion</c> source, and the diffusion transformers FLUX.1, SD3 / SD3.5 and Chroma (diffusers
/// folders, original-name single files and transformer-only GGUF files) translated into the canonical <c>flux</c>, <c>sd3</c>,
/// <c>chroma</c> and <c>zimage</c> sources; plus the stand-alone component files that come with transformer-only checkpoints (CLIP and T5 text encoders,
/// VAEs), which open as component sources and are found beside the transformer (<see cref="FindCompanions"/>). The
/// <c>LatentDiffusionAdapter</c>, <c>FluxAdapter</c>, <c>Sd3Adapter</c> and <c>ChromaAdapter</c> of Sharplight.Core build the
/// image-generation models.
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
    /// Also claims single-file checkpoints of the diffusion transformers that are not supported yet (Qwen-Image, video DiTs, ...), which
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
        bool supported = (kind == FluxConfig.Architecture && family?.StartsWith("FLUX", StringComparison.Ordinal) == true)
            || (kind == Sd3Config.Architecture && family?.StartsWith("Stable Diffusion 3", StringComparison.Ordinal) == true)
            || kind == ChromaConfig.Architecture || kind == ZImageConfig.Architecture || kind == WanConfig.Architecture;
        if (family is not null && !supported)
        {
            throw new NotSupportedException($"Not supported yet: {family}. Sharplight generates images with the Stable Diffusion UNet models "
                + $"(SD 1.x, SD 2.x, SDXL), SD3 / SD3.5, FLUX.1, Chroma and Z-Image, and video with Wan; {Path.GetFileName(path)} is another diffusion-transformer model.");
        }

        return kind switch
        {
            FluxConfig.Architecture or ChromaConfig.Architecture => FluxCheckpoint.Open(path),
            Sd3Config.Architecture => Sd3Checkpoint.Open(path),
            ZImageConfig.Architecture => ZImageCheckpoint.Open(path),
            WanConfig.Architecture => WanCheckpoint.Open(path),
            ComponentCheckpoint.ClipArchitecture or ComponentCheckpoint.AutoencoderArchitecture or ComponentCheckpoint.T5Architecture => ComponentCheckpoint.Open(path),
            _ => SingleFileCheckpoint.Open(path),
        };
    }

    /// <summary>The companion files a transformer-only FLUX / SD3 file needs (<see cref="DiffusionCompanions.Describe"/>).</summary>
    /// <param name="path">The model file.</param>
    public IReadOnlyList<CompanionFile> DescribeCompanions(string path)
    {
        string? kind = File.Exists(path) ? SingleFileCheckpoint.Kind(path) : null;
        return kind switch
        {
            FluxConfig.Architecture => DiffusionCompanions.Describe(CompanionFamily.Flux),
            ChromaConfig.Architecture => DiffusionCompanions.Describe(CompanionFamily.Chroma),
            ZImageConfig.Architecture => DiffusionCompanions.Describe(CompanionFamily.ZImage),
            Sd3Config.Architecture => DiffusionCompanions.Describe(CompanionFamily.Sd3),
            _ => [],
        };
    }

    /// <summary>The text encoders and VAE a transformer-only FLUX / SD3 file needs, found beside it (<see cref="DiffusionCompanions"/>).</summary>
    /// <param name="path">The model file.</param>
    public IReadOnlyDictionary<string, string> FindCompanions(string path)
    {
        string? kind = File.Exists(path) ? SingleFileCheckpoint.Kind(path) : null;
        return kind switch
        {
            FluxConfig.Architecture => DiffusionCompanions.Find(path, CompanionFamily.Flux),
            ChromaConfig.Architecture => DiffusionCompanions.Find(path, CompanionFamily.Chroma),
            ZImageConfig.Architecture => DiffusionCompanions.Find(path, CompanionFamily.ZImage),
            Sd3Config.Architecture => DiffusionCompanions.Find(path, CompanionFamily.Sd3),
            _ => new Dictionary<string, string>(),
        };
    }

    /// <inheritdoc/>
    public PipelineSupport? ClassifyPipeline(string pipelineClass, IReadOnlyCollection<string> components)
    {
        ArgumentNullException.ThrowIfNull(pipelineClass);
        ArgumentNullException.ThrowIfNull(components);
        if (!pipelineClass.EndsWith("Pipeline", StringComparison.Ordinal))
        {
            return null;
        }

        return pipelineClass switch
        {
            "StableDiffusionPipeline" or "StableDiffusionImg2ImgPipeline" or "StableDiffusionInpaintPipeline" or "StableDiffusionXLPipeline"
                or "StableDiffusionXLImg2ImgPipeline" or "StableDiffusionXLInpaintPipeline" or "FluxPipeline" or "WanPipeline" or "WanImageToVideoPipeline" or "LTXPipeline"
                or "StableDiffusion3Pipeline" or "StableDiffusion3Img2ImgPipeline" or "StableDiffusion3InpaintPipeline" or "ChromaPipeline"
                or "ChromaImg2ImgPipeline" or "ZImagePipeline" or "ZImageImg2ImgPipeline"
                => components.Contains("unet") || components.Contains("transformer") ? new PipelineSupport(true) : new PipelineSupport(false, $"Not supported yet: {pipelineClass} has no UNet or transformer."),
            _ => new PipelineSupport(false, $"Not supported yet: {Describe(pipelineClass)}. Sharplight loads the Stable Diffusion 1.x / 2.x / XL / 3, FLUX.1, Chroma, Z-Image, Wan 2.1 / 2.2 and LTX-Video pipelines."),
        };
    }

    private static string Describe(string pipelineClass)
    {
        string name = pipelineClass[..^"Pipeline".Length];
        return name switch
        {
            "AuraFlow" => "AuraFlow",
            "PixArtSigma" or "PixArtAlpha" => "PixArt",
            "HunyuanVideo" => "HunyuanVideo",
            "CogVideoX" => "CogVideoX",
            "LTX" => "LTX-Video",
            "Kandinsky" or "Kandinsky3" => "Kandinsky",
            _ => $"the {pipelineClass} diffusers pipeline",
        };
    }
}
