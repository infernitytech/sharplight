using Sharplight.Import.Gguf;
using Sharplight.Import.HuggingFace;
using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// Reads an original-layout (CompVis / Stability "LDM") Stable Diffusion checkpoint in one file: a single-file <c>.safetensors</c>
/// checkpoint, or a GGUF file as written by stable-diffusion.cpp (the same tensor names, linear weights quantized). Such files
/// carry no configuration, so the hyper-parameters are inferred from the tensor shapes and the family defaults (SD 1.x).
/// </summary>
internal static class SingleFileCheckpoint
{
    private const string UNetPrefix = "model.diffusion_model.";
    private const string VaePrefix = "first_stage_model.";
    private const string OpenAiClipPrefix = "cond_stage_model.transformer.";
    private const string OpenClipPrefix = "cond_stage_model.model.";
    private const string SdxlPrefix = "conditioner.embedders.";

    /// <summary>Whether <paramref name="path"/> is a safetensors or GGUF file holding a latent-diffusion UNet.</summary>
    /// <param name="path">A file.</param>
    public static bool IsCheckpoint(string path)
    {
        if (!File.Exists(path))
        {
            return false;
        }

        try
        {
            return TensorNames(path).Any(n => n.StartsWith(UNetPrefix, StringComparison.Ordinal));
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

    /// <summary>Opens the checkpoint as a canonical source.</summary>
    /// <param name="path">The file.</param>
    /// <exception cref="NotSupportedException">The checkpoint is not an SD 1.x checkpoint.</exception>
    public static ModelSource Open(string path)
    {
        var tensors = new CheckpointTensors();
        try
        {
            tensors.Add(path);
            var names = tensors.Tensors.Keys.ToList();
            if (names.Any(n => n.StartsWith(OpenClipPrefix, StringComparison.Ordinal) || n.StartsWith(SdxlPrefix, StringComparison.Ordinal)))
            {
                throw new NotSupportedException($"'{path}': only SD 1.x single-file checkpoints (OpenAI CLIP text encoder) are supported so far.");
            }

            var weights = new CanonicalWeights();
            AddUNet(tensors, names, weights);
            AddVae(tensors, names, weights);
            foreach (string name in names.Where(n => n.StartsWith(OpenAiClipPrefix, StringComparison.Ordinal)))
            {
                DiffusersNames.AddClipText(name[OpenAiClipPrefix.Length..], tensors.Tensors[name], LatentDiffusionSource.TextEncoderPrefix, weights);
            }

            const int sampleSize = 64;
            var unet = UNetConfig.FromWeights(weights.Weights, sampleSize);
            var vae = VaeConfig.FromWeights(weights.Weights, 0.18215f, LatentDiffusionSource.StableDiffusion1RgbFactors);
            var text = ClipTextConfig.FromWeights(weights.Weights, LatentDiffusionSource.TextEncoderPrefix, "quick_gelu");
            var scheduler = new SchedulerSettings();
            string format = CheckpointTensors.IsGguf(path) ? "gguf" : "safetensors";
            return LatentDiffusionSource.Create(format, Path.GetFileNameWithoutExtension(path), path, weights, unet, vae, text,
                ClipVocabulary.OpenAi(), scheduler, tensors.Resources);
        }
        catch
        {
            tensors.DisposeAll();
            throw;
        }
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

    /// <summary>The tensor names of a safetensors or GGUF file (header only).</summary>
    private static IEnumerable<string> TensorNames(string path)
    {
        if (CheckpointTensors.IsGguf(path))
        {
            using var gguf = GgufFile.Open(path);
            return [.. gguf.Tensors.Select(t => t.Name)];
        }

        if (!path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
        {
            return [];
        }

        using var file = SafetensorsFile.Open(path);
        return [.. file.Tensors.Keys];
    }
}
