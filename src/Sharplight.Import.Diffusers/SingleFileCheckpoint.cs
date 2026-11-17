using Sharplight.Import.Gguf;
using Sharplight.Import.HuggingFace;
using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Reads an original-layout (CompVis / Stability "LDM") Stable Diffusion checkpoint in one file: a single-file <c>.safetensors</c>
/// checkpoint, a PyTorch <c>.ckpt</c> (read without executing its pickle), or a GGUF file as written by stable-diffusion.cpp (the same tensor names, linear weights quantized). Such files
/// carry no configuration, so the hyper-parameters are inferred from the tensor shapes and the family defaults (SD 1.x: OpenAI
/// CLIP text encoder; SD 2.x: OpenCLIP ViT-H; SDXL: both, CLIP-L and OpenCLIP bigG).
/// </summary>
internal static partial class SingleFileCheckpoint
{
    private const string UNetPrefix = "model.diffusion_model.";
    private const string VaePrefix = "first_stage_model.";
    private const string OpenAiClipPrefix = "cond_stage_model.transformer.";
    private const string OpenClipPrefix = "cond_stage_model.model.";
    private const string SdxlPrefix = "conditioner.embedders.";
    private const string SdxlClipPrefix = "conditioner.embedders.0.transformer.";
    private const string SdxlOpenClipPrefix = "conditioner.embedders.1.model.";
    private const string SdCppXlOpenClipPrefix = "cond_stage_model.1.transformer.";
    private const string SdCppTextProjection = "text_model.text_projection";

    /// <summary>Whether <paramref name="path"/> is a safetensors, GGUF or PyTorch checkpoint holding a latent-diffusion UNet.</summary>
    /// <param name="path">A file.</param>
    public static bool IsCheckpoint(string path) => Kind(path) == LatentDiffusionSource.Architecture;

    /// <summary>
    /// What a checkpoint file holds: <c>latent_diffusion</c> (a Stable Diffusion UNet checkpoint), <c>flux</c> / <c>chroma</c> / <c>sd3</c> (an
    /// original-layout FLUX / Chroma / SD3 transformer), a stand-alone component (<see cref="ComponentCheckpoint.Classify"/>), or null (not a diffusion file, or unreadable).
    /// </summary>
    /// <param name="path">A file.</param>
    public static string? Kind(string path)
    {
        if (!File.Exists(path))
        {
            return null;
        }

        try
        {
            var names = TensorNames(path);
            if (Krea2Checkpoint.TransformerPrefix(names) is not null)
            {
                return Krea2Config.Architecture;
            }

            if (Flux2Checkpoint.TransformerPrefix(names) is not null)
            {
                return Flux2Config.Architecture;
            }

            if (FluxCheckpoint.TransformerPrefix(names) is { } fluxPrefix)
            {
                return names.Contains(fluxPrefix + "distilled_guidance_layer.in_proj.weight") ? ChromaConfig.Architecture : FluxConfig.Architecture;
            }

            if (Sd3Checkpoint.TransformerPrefix(names) is not null)
            {
                return Sd3Config.Architecture;
            }

            if (ZImageCheckpoint.TransformerPrefix(names) is not null)
            {
                return ZImageConfig.Architecture;
            }

            if (WanCheckpoint.TransformerPrefix(names) is not null)
            {
                return WanConfig.Architecture;
            }

            if (HunyuanVideoCheckpoint.TransformerPrefix(names) is not null)
            {
                return HunyuanVideoFolder.Architecture;
            }

            if (LtxCheckpoint.TransformerPrefix(names) is not null || LtxCheckpoint.IsVae(names))
            {
                return LtxFolder.Architecture;
            }

            if (QwenImageCheckpoint.TransformerPrefix(names) is not null)
            {
                return QwenImageConfig.Architecture;
            }

            if (QwenImage21Checkpoint.TransformerPrefix(names) is not null)
            {
                return QwenImage21Config.Architecture;
            }

            if (names.Any(n => n.StartsWith(UNetPrefix, StringComparison.Ordinal)))
            {
                return LatentDiffusionSource.Architecture;
            }

            return path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase) ? ComponentCheckpoint.Classify(names) : null;
        }
        catch (IOException)
        {
            return null;
        }
        catch (InvalidDataException)
        {
            return null;
        }
        catch (Sharplight.Import.PyTorch.UnsafePickleException)
        {
            return null;
        }
        catch (NotSupportedException)
        {
            return null;
        }
    }

    /// <summary>Opens the checkpoint as a canonical source.</summary>
    /// <param name="path">The file.</param>
    /// <exception cref="NotSupportedException">The checkpoint is not an SD 1.x / 2.x / XL checkpoint.</exception>
    public static ModelSource Open(string path)
    {
        var tensors = new CheckpointTensors();
        try
        {
            tensors.Add(path);
            var names = tensors.Tensors.Keys.ToList();
            var weights = new CanonicalWeights();
            AddUNet(tensors, names, weights);
            AddVae(tensors, names, weights);
            var missing = new List<string>();
            if (!names.Any(n => n.StartsWith(VaePrefix, StringComparison.Ordinal)))
            {
                missing.Add("the VAE ('first_stage_model.*')");
            }

            if (!names.Any(n => n.StartsWith("cond_stage_model.", StringComparison.Ordinal) || n.StartsWith(SdxlPrefix, StringComparison.Ordinal)))
            {
                missing.Add("the text encoder ('cond_stage_model.*' or 'conditioner.embedders.*')");
            }

            if (missing.Count > 0)
            {
                throw new NotSupportedException($"{path}: the file holds the UNet but not {string.Join(" or ", missing)}. Separate VAE and text "
                    + "encoder files are not supported yet; use a full single-file checkpoint (UNet, VAE and text encoders in one file).");
            }

            string format = CheckpointTensors.IsGguf(path) ? "gguf" : TorchCheckpoints.HasExtension(path) ? "pytorch" : "safetensors";
            if (names.Any(n => n.StartsWith(SdxlPrefix, StringComparison.Ordinal)))
            {
                var second = OpenClipNames.ToTransformers(tensors.Tensors, SdxlOpenClipPrefix, droppedLayers: 0, projection: true);
                return OpenXl(path, format, tensors, weights, ClipTensors(tensors, names, SdxlClipPrefix), second);
            }

            if (names.Any(n => n.StartsWith(SdCppXlOpenClipPrefix, StringComparison.Ordinal)))
            {
                // stable-diffusion.cpp's SDXL layout: both encoders under transformers names. sd.cpp renames the bigG projection without
                // transposing it, so its orientation is the source checkpoint's: transformers' (out × in) when the file was converted from a
                // diffusers folder, OpenCLIP's transpose when converted from an original checkpoint (verified against the diffusers
                // weights for gpustack's and OlegSkutte's files). The diffusers conversion leaves a mark: the upsampler of an up level with
                // attention sits at index 1 beside the attention block (see LdmNames.UNet).
                bool fromDiffusers = names.Any(n => n.StartsWith(UNetPrefix + "output_blocks.", StringComparison.Ordinal) && n.EndsWith(".1.conv.weight", StringComparison.Ordinal)
                    && names.Contains(n[..^"conv.weight".Length] + "proj_in.weight"));
                var second = ClipTensors(tensors, names, SdCppXlOpenClipPrefix).Select(pair => pair.Name != SdCppTextProjection ? pair
                    : ("text_projection.weight", fromDiffusers ? pair.Weight : OpenClipNames.Transposed(pair.Weight, "text_projection.weight")));
                return OpenXl(path, format, tensors, weights, ClipTensors(tensors, names, OpenAiClipPrefix), second);
            }

            bool openClip = names.Any(n => n.StartsWith(OpenClipPrefix, StringComparison.Ordinal));
            if (openClip)
            {
                // SD 2.x: OpenCLIP ViT-H, conditioned on the penultimate layer.
                foreach (var (name, weight) in OpenClipNames.ToTransformers(tensors.Tensors, OpenClipPrefix, droppedLayers: 1))
                {
                    DiffusersNames.AddClipText(name, weight, LatentDiffusionSource.TextEncoderPrefix, weights);
                }
            }
            else
            {
                foreach (string name in names.Where(n => n.StartsWith(OpenAiClipPrefix, StringComparison.Ordinal)))
                {
                    DiffusersNames.AddClipText(name[OpenAiClipPrefix.Length..], tensors.Tensors[name], LatentDiffusionSource.TextEncoderPrefix, weights);
                }
            }

            var (sampleSize, scheduler) = openClip ? Sd2Defaults(path) : (64, new SchedulerSettings());
            var unet = UNetConfig.FromWeights(weights.Weights, sampleSize);
            if (unet.ClassEmbeddingInput > 0)
            {
                throw new NotSupportedException($"{path}: the UNet is an SDXL UNet (conditioned on a pooled text embedding), but the file holds no "
                    + "second text encoder (OpenCLIP bigG: 'conditioner.embedders.1.*' or 'cond_stage_model.1.*'). Use a checkpoint that contains both "
                    + "SDXL text encoders and the VAE.");
            }

            var vae = VaeConfig.FromWeights(weights.Weights, 0.18215f, LatentDiffusionSource.StableDiffusion1RgbFactors);
            var text = ClipTextConfig.FromWeights(weights.Weights, LatentDiffusionSource.TextEncoderPrefix, openClip ? "gelu" : "quick_gelu");
            return LatentDiffusionSource.Create(format, Path.GetFileNameWithoutExtension(path), path, weights, unet, vae, text,
                ClipVocabulary.OpenAi(openClip ? "!" : ClipVocabulary.EndToken), scheduler, tensors.Resources);
        }
        catch
        {
            tensors.DisposeAll();
            throw;
        }
    }

    /// <summary>The tensors under <paramref name="prefix"/>, named without it.</summary>
    private static IEnumerable<(string Name, WeightInfo Weight)> ClipTensors(CheckpointTensors tensors, List<string> names, string prefix)
        => names.Where(n => n.StartsWith(prefix, StringComparison.Ordinal)).Select(n => (n[prefix.Length..], tensors.Tensors[n]));

    /// <summary>
    /// SDXL: the OpenAI CLIP-L encoder (SGM layout: <c>conditioner.embedders.0.transformer.</c>; stable-diffusion.cpp:
    /// <c>cond_stage_model.transformer.</c>) and the OpenCLIP bigG encoder with its text projection (SGM:
    /// <c>conditioner.embedders.1.model.</c>; stable-diffusion.cpp: <c>cond_stage_model.1.transformer.</c> under transformers names),
    /// given as transformers names, both read at the penultimate layer without the final norm; the UNet's added conditioning is the
    /// pooled embedding plus six 256-wide size embeddings; 1024×1024, Euler.
    /// </summary>
    private static ModelSource OpenXl(string path, string format, CheckpointTensors tensors, CanonicalWeights weights,
        IEnumerable<(string Name, WeightInfo Weight)> first, IEnumerable<(string Name, WeightInfo Weight)> second)
    {
        foreach (var (name, weight) in first)
        {
            DiffusersNames.AddClipText(name, weight, LatentDiffusionSource.TextEncoderPrefix, weights);
        }

        foreach (var (name, weight) in second)
        {
            DiffusersNames.AddClipText(name, weight, LatentDiffusionSource.SecondTextEncoderPrefix, weights);
        }

        const int sizeConditions = 6;
        var (sampleSize, scheduler) = XlDefaults(path);
        var unet = UNetConfig.FromWeights(weights.Weights, sampleSize);
        int pooledWidth = (int)weights.Weights[LatentDiffusionSource.SecondTextEncoderPrefix + "projection.weight"].Shape[0].Fixed;
        unet = unet with { AdditionTimeDim = (unet.ClassEmbeddingInput - pooledWidth) / sizeConditions };
        var vae = VaeConfig.FromWeights(weights.Weights, 0.13025f, LatentDiffusionSource.StableDiffusion1RgbFactors);
        var text = ClipTextConfig.FromWeights(weights.Weights, LatentDiffusionSource.TextEncoderPrefix, "quick_gelu") with { ClipSkip = 2, FinalNorm = false };
        var secondText = ClipTextConfig.FromWeights(weights.Weights, LatentDiffusionSource.SecondTextEncoderPrefix, "gelu")
            with { ClipSkip = 2, FinalNorm = false };
        return LatentDiffusionSource.Create(format, Path.GetFileNameWithoutExtension(path), path, weights, unet, vae, text, ClipVocabulary.OpenAi(),
            scheduler, tensors.Resources, secondText, ClipVocabulary.OpenAi("!").PadId, zeroEmptyNegative: true);
    }

    /// <summary>
    /// The native latent size and sampling defaults of an SDXL checkpoint, which its tensors do not reveal. The distilled variants are
    /// recognized by their published file names, as their diffusers configurations set them up: SDXL-Turbo (512×512, Euler ancestral on
    /// trailing timesteps, 4 steps, no guidance) and SDXL-Lightning (1024×1024, Euler on trailing timesteps, the step count of the file,
    /// no guidance; the 1-step "x0" file predicts the clean sample). Everything else is the base model's setup: 1024×1024, Euler.
    /// </summary>
    private static (int SampleSize, SchedulerSettings Scheduler) XlDefaults(string path)
    {
        string name = Path.GetFileName(path);
        if (name.Contains("turbo", StringComparison.OrdinalIgnoreCase))
        {
            return (64, new SchedulerSettings
            {
                Sampler = "euler-ancestral",
                TimestepSpacing = "trailing",
                StepsOffset = 0,
                DefaultSteps = 4,
                DefaultGuidance = 0f,
            });
        }

        var lightning = LightningSteps().Match(name);
        if (name.Contains("lightning", StringComparison.OrdinalIgnoreCase) && lightning.Success)
        {
            return (128, new SchedulerSettings
            {
                Sampler = "euler",
                TimestepSpacing = "trailing",
                StepsOffset = 0,
                PredictionType = name.Contains("x0", StringComparison.OrdinalIgnoreCase) ? "sample" : "epsilon",
                DefaultSteps = int.Parse(lightning.Groups[1].ValueSpan, System.Globalization.CultureInfo.InvariantCulture),
                DefaultGuidance = 0f,
            });
        }

        return (128, new SchedulerSettings { Sampler = "euler", TimestepSpacing = "leading" });
    }

    [System.Text.RegularExpressions.GeneratedRegex(@"(\d+)[-_]?step", System.Text.RegularExpressions.RegexOptions.IgnoreCase)]
    private static partial System.Text.RegularExpressions.Regex LightningSteps();

    /// <summary>
    /// The native latent size and noise schedule of an SD 2.x checkpoint, which its tensors do not reveal: the 768×768 models
    /// predict v, the 512×512 "base" models noise. Files named like the official 512 / base releases get the latter.
    /// </summary>
    private static (int SampleSize, SchedulerSettings Scheduler) Sd2Defaults(string path)
    {
        string name = Path.GetFileName(path);
        bool epsilon = name.Contains("512", StringComparison.Ordinal) || name.Contains("base", StringComparison.OrdinalIgnoreCase);
        var scheduler = new SchedulerSettings
        {
            PredictionType = epsilon ? "epsilon" : "v_prediction",
            ClipSample = false,
            Sampler = "ddim",
        };
        return (epsilon ? 64 : 96, scheduler);
    }

    private static void AddUNet(CheckpointTensors tensors, List<string> names, CanonicalWeights weights)
    {
        var unetNames = names.Where(n => n.StartsWith(UNetPrefix, StringComparison.Ordinal)).Select(n => n[UNetPrefix.Length..]).ToList();
        foreach (var (ldm, diffusers) in LdmNames.UNet(unetNames))
        {
            DiffusersNames.AddUNet(diffusers, tensors.Tensors[UNetPrefix + ldm], weights);
        }
    }

    private static void AddVae(CheckpointTensors tensors, List<string> names, CanonicalWeights weights)
    {
        var vaeNames = names.Where(n => n.StartsWith(VaePrefix, StringComparison.Ordinal) && !n.Contains(".loss.", StringComparison.Ordinal))
            .Select(n => n[VaePrefix.Length..]).ToList();
        foreach (var (ldm, diffusers) in LdmNames.Vae(vaeNames))
        {
            DiffusersNames.AddVae(diffusers, tensors.Tensors[VaePrefix + ldm], weights);
        }
    }

    /// <summary>The tensor names of a safetensors, GGUF or PyTorch checkpoint file (header / pickle only).</summary>
    internal static IReadOnlyCollection<string> TensorNames(string path)
    {
        if (CheckpointTensors.IsGguf(path))
        {
            using var gguf = GgufFile.Open(path);
            return [.. gguf.Tensors.Select(t => t.Name)];
        }

        if (TorchCheckpoints.HasExtension(path))
        {
            using var checkpoint = TorchCheckpoints.Open(path);
            return [.. checkpoint.Tensors.Keys];
        }

        if (!path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
        {
            return [];
        }

        using var file = SafetensorsFile.Open(path);
        return [.. file.Tensors.Keys];
    }
}
