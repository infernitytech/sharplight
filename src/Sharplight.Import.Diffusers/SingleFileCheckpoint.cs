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
internal static class SingleFileCheckpoint
{
    private const string UNetPrefix = "model.diffusion_model.";
    private const string VaePrefix = "first_stage_model.";
    private const string OpenAiClipPrefix = "cond_stage_model.transformer.";
    private const string OpenClipPrefix = "cond_stage_model.model.";
    private const string SdxlPrefix = "conditioner.embedders.";
    private const string SdxlClipPrefix = "conditioner.embedders.0.transformer.";
    private const string SdxlOpenClipPrefix = "conditioner.embedders.1.model.";

    /// <summary>Whether <paramref name="path"/> is a safetensors, GGUF or PyTorch checkpoint holding a latent-diffusion UNet.</summary>
    /// <param name="path">A file.</param>
    public static bool IsCheckpoint(string path) => Kind(path) == LatentDiffusionSource.Architecture;

    /// <summary>
    /// What a checkpoint file holds: <c>latent_diffusion</c> (a Stable Diffusion UNet checkpoint), <c>flux</c> (an original-layout FLUX
    /// transformer), a stand-alone component (<see cref="ComponentCheckpoint.Classify"/>), or null (not a diffusion file, or unreadable).
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
            if (FluxCheckpoint.TransformerPrefix(names) is not null)
            {
                return FluxConfig.Architecture;
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
            string format = CheckpointTensors.IsGguf(path) ? "gguf" : TorchCheckpoints.HasExtension(path) ? "pytorch" : "safetensors";
            if (names.Any(n => n.StartsWith(SdxlPrefix, StringComparison.Ordinal)))
            {
                return OpenXl(path, format, tensors, names, weights);
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

    /// <summary>
    /// SDXL (SGM layout): the OpenAI CLIP-L encoder under <c>conditioner.embedders.0.transformer.</c> and the OpenCLIP bigG encoder
    /// with its text projection under <c>conditioner.embedders.1.model.</c>, both read at the penultimate layer without the final
    /// norm; the UNet's added conditioning is the pooled embedding plus six 256-wide size embeddings; 1024×1024, Euler.
    /// </summary>
    private static ModelSource OpenXl(string path, string format, CheckpointTensors tensors, List<string> names, CanonicalWeights weights)
    {
        foreach (string name in names.Where(n => n.StartsWith(SdxlClipPrefix, StringComparison.Ordinal)))
        {
            DiffusersNames.AddClipText(name[SdxlClipPrefix.Length..], tensors.Tensors[name], LatentDiffusionSource.TextEncoderPrefix, weights);
        }

        foreach (var (name, weight) in OpenClipNames.ToTransformers(tensors.Tensors, SdxlOpenClipPrefix, droppedLayers: 0, projection: true))
        {
            DiffusersNames.AddClipText(name, weight, LatentDiffusionSource.SecondTextEncoderPrefix, weights);
        }

        const int sizeConditions = 6;
        var unet = UNetConfig.FromWeights(weights.Weights, 128);
        int pooledWidth = (int)weights.Weights[LatentDiffusionSource.SecondTextEncoderPrefix + "projection.weight"].Shape[0].Fixed;
        unet = unet with { AdditionTimeDim = (unet.ClassEmbeddingInput - pooledWidth) / sizeConditions };
        var vae = VaeConfig.FromWeights(weights.Weights, 0.13025f, LatentDiffusionSource.StableDiffusion1RgbFactors);
        var text = ClipTextConfig.FromWeights(weights.Weights, LatentDiffusionSource.TextEncoderPrefix, "quick_gelu") with { ClipSkip = 2, FinalNorm = false };
        var secondText = ClipTextConfig.FromWeights(weights.Weights, LatentDiffusionSource.SecondTextEncoderPrefix, "gelu")
            with { ClipSkip = 2, FinalNorm = false };
        var scheduler = new SchedulerSettings { Sampler = "euler", TimestepSpacing = "leading" };
        return LatentDiffusionSource.Create(format, Path.GetFileNameWithoutExtension(path), path, weights, unet, vae, text, ClipVocabulary.OpenAi(),
            scheduler, tensors.Resources, secondText, ClipVocabulary.OpenAi("!").PadId, zeroEmptyNegative: true);
    }

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
    private static IReadOnlyCollection<string> TensorNames(string path)
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
