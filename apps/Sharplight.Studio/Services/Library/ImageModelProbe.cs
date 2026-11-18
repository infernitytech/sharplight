using System.Text;
using System.Text.Json;
using Sharplight.Import.Gguf;
using Sharplight.Models;
using Sharplight.Studio.Models;

namespace Sharplight.Studio.Services.Library;

/// <summary>
/// Recognizes the image (and Wan video) models of the library from their content (or, where a file cannot be read cheaply, their name): Stable
/// Diffusion and FLUX pipelines (a diffusers folder, a single-file checkpoint, a stable-diffusion.cpp / city96 GGUF; FLUX transformer-only
/// files need companion files, <see cref="NeedsCompanions"/>) and the Hugging Face vision
/// checkpoints (image classification, image embedding, object detection, depth estimation) by the architecture of their
/// <c>config.json</c>. Anything else gives null.
/// </summary>
internal static class ImageModelProbe
{
    private const string UNetPrefix = "model.diffusion_model.";
    private const string FluxTransformerTensor = "double_blocks.0.img_attn.qkv.weight";
    private const string Sd3TransformerTensor = "joint_blocks.0.x_block.attn.qkv.weight";
    private const string ZImageTransformerTensor = "noise_refiner.0.attention.qkv.weight";
    private const string QwenImageTransformerTensor = "transformer_blocks.0.img_mod.1.weight";
    private const string QwenImage21TransformerTensor = "txt_in.text_norm.weight";
    private const string Krea2TransformerTensor = "txtfusion.projector.weight";
    private const long MaxSafetensorsHeader = 256L << 20;

    private static readonly HashSet<string> s_diffusionTransformerArchitectures = new(StringComparer.OrdinalIgnoreCase)
    {
        "flux", "sd3", "krea2", "aura", "auraflow", "chroma", "hidream", "lumina2", "qwen_image", "wan", "hyvid", "ltxv", "cosmos",
    };

    private static readonly string[] s_upscalerNameHints = ["esrgan", "swinir", "realesr", "upscal", "super-resolution", "superres"];

    // Tensor names of the RRDBNet (Real-ESRGAN) and SwinIR super-resolution networks.
    private static readonly string[] s_upscalerTensors = ["body.0.rdb1.conv1.weight", "RRDB_trunk.0.RDB1", "conv_before_upsample"];

    private static readonly string[] s_checkpointNameHints = ["stable-diffusion", "stable_diffusion", "sd-", "sd_", "sd1", "sd2", "sdxl", "v1-5", "v2-1", "dreamshaper", "diffusion"];

    /// <summary>The kind of the model file or folder, or null when it is not a recognized image model.</summary>
    /// <param name="path">A model file or folder.</param>
    public static string? KindOf(string path)
    {
        if (Directory.Exists(path))
        {
            if (IsDiffusersFolder(path))
            {
                return IsVideoDiffusersFolder(path) ? ModelKinds.VideoGeneration : ModelKinds.ImageGeneration;
            }

            if (File.Exists(Path.Combine(path, "config.json")))
            {
                return KindOfConfig(Path.Combine(path, "config.json"));
            }

            // A folder holding a Real-ESRGAN / SwinIR checkpoint (.pth, or a named .safetensors).
            return Directory.EnumerateFiles(path, "*.pth").Concat(Directory.EnumerateFiles(path, "*.safetensors")).Any(file => KindOf(file) == ModelKinds.ImageUpscaling) ? ModelKinds.ImageUpscaling : null;
        }

        if (path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
        {
            if (IsLora(path))
            {
                return ModelKindCatalog.Lora;
            }

            if (SafetensorsHeaderContains(path, UNetPrefix) || SafetensorsHeaderContains(path, FluxTransformerTensor)
                || SafetensorsHeaderContains(path, Sd3TransformerTensor) || SafetensorsHeaderContains(path, ZImageTransformerTensor)
                || SafetensorsHeaderContains(path, QwenImageTransformerTensor) || SafetensorsHeaderContains(path, QwenImage21TransformerTensor)
                || SafetensorsHeaderContains(path, Krea2TransformerTensor))
            {
                return ModelKinds.ImageGeneration;
            }

            return s_upscalerTensors.Any(tensor => SafetensorsHeaderContains(path, tensor)) ? ModelKinds.ImageUpscaling : null;
        }

        // Real-ESRGAN and SwinIR are distributed as pickled .pth files, which the name recognizes.
        if (path.EndsWith(".pth", StringComparison.OrdinalIgnoreCase))
        {
            string name = Path.GetFileName(path);
            return s_upscalerNameHints.Any(hint => name.Contains(hint, StringComparison.OrdinalIgnoreCase)) ? ModelKinds.ImageUpscaling : null;
        }

        if (path.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
        {
            return GgufHasDiffusionModel(path) ? ModelKinds.ImageGeneration : null;
        }

        // A pickled checkpoint cannot be read without the engine's restricted unpickler, so the name decides.
        if (path.EndsWith(".ckpt", StringComparison.OrdinalIgnoreCase))
        {
            string name = Path.GetFileName(path);
            return s_checkpointNameHints.Any(hint => name.Contains(hint, StringComparison.OrdinalIgnoreCase)) ? ModelKinds.ImageGeneration : null;
        }

        return null;
    }

    /// <summary>
    /// Whether a <c>.safetensors</c> file is a LoRA adapter for an image model: kohya-ss (<c>.lora_down.weight</c> / <c>.lora_up.weight</c>) or
    /// diffusers / PEFT (<c>.lora_A.weight</c> / <c>.lora_B.weight</c>, also <c>.lora.down.weight</c>) tensor names. A full checkpoint has none.
    /// </summary>
    /// <param name="path">A <c>.safetensors</c> file.</param>
    public static bool IsLora(string path)
        => SafetensorsHeaderContains(path, ".lora_down.weight") || SafetensorsHeaderContains(path, ".lora_A.weight")
           || SafetensorsHeaderContains(path, ".lora_A.default.weight") || SafetensorsHeaderContains(path, ".lora.down.weight");

    /// <summary>
    /// Whether an image model file holds only a FLUX, Chroma, SD3, Z-Image or Qwen-Image transformer (original tensor names, as city96 / stable-diffusion.cpp GGUF files
    /// and the official single files are distributed), so its text encoders and VAE come as companion files.
    /// </summary>
    /// <param name="path">A model file.</param>
    public static bool NeedsCompanions(string path)
    {
        if (path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
        {
            return (SafetensorsHeaderContains(path, FluxTransformerTensor) || SafetensorsHeaderContains(path, Sd3TransformerTensor)
                    || SafetensorsHeaderContains(path, ZImageTransformerTensor) || SafetensorsHeaderContains(path, QwenImageTransformerTensor)
                    || SafetensorsHeaderContains(path, QwenImage21TransformerTensor) || SafetensorsHeaderContains(path, Krea2TransformerTensor))
                && !SafetensorsHeaderContains(path, "text_encoders.");
        }

        if (!path.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
        {
            return false;
        }

        try
        {
            using var file = GgufFile.Open(path);
            return file.Tensors.Any(tensor => tensor.Name.EndsWith(FluxTransformerTensor, StringComparison.Ordinal)
                || tensor.Name.EndsWith(Sd3TransformerTensor, StringComparison.Ordinal) || tensor.Name.EndsWith(ZImageTransformerTensor, StringComparison.Ordinal)
                || tensor.Name.EndsWith(QwenImageTransformerTensor, StringComparison.Ordinal) || tensor.Name.EndsWith(QwenImage21TransformerTensor, StringComparison.Ordinal)
                || tensor.Name.EndsWith(Krea2TransformerTensor, StringComparison.Ordinal));
        }
        catch (Exception error) when (error is IOException or InvalidDataException or NotSupportedException or UnauthorizedAccessException)
        {
            return false;
        }
    }

    /// <summary>
    /// A video pipeline (Wan, LTX-Video, HunyuanVideo): the transformer is a Wan transformer, or the pipeline's own <c>model_index.json</c> names a
    /// video pipeline.
    /// </summary>
    private static bool IsVideoDiffusersFolder(string path)
        => ClassNameOf(Path.Combine(path, "transformer", "config.json")) is "WanTransformer3DModel"
           || ClassNameOf(Path.Combine(path, "model_index.json")) is { } pipeline && s_videoPipelinePrefixes.Any(prefix => pipeline.StartsWith(prefix, StringComparison.Ordinal));

    private static readonly string[] s_videoPipelinePrefixes = ["Wan", "LTX", "HunyuanVideo"];

    /// <summary>The <c>_class_name</c> of a diffusers JSON file, or null when the file is missing, unreadable or names none.</summary>
    private static string? ClassNameOf(string file)
    {
        if (!File.Exists(file))
        {
            return null;
        }

        try
        {
            using var document = JsonDocument.Parse(File.ReadAllBytes(file));
            return document.RootElement.TryGetProperty("_class_name", out var name) && name.ValueKind == JsonValueKind.String ? name.GetString() : null;
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or JsonException)
        {
            return null;
        }
    }

    private static bool IsDiffusersFolder(string path)
        => File.Exists(Path.Combine(path, "model_index.json"))
           || ((File.Exists(Path.Combine(path, "unet", "config.json")) || File.Exists(Path.Combine(path, "transformer", "config.json")))
               && File.Exists(Path.Combine(path, "vae", "config.json")));

    /// <summary>The header of a safetensors file is JSON naming every tensor, so the check reads the header only.</summary>
    private static bool SafetensorsHeaderContains(string path, string text)
    {
        try
        {
            using var stream = File.OpenRead(path);
            Span<byte> length = stackalloc byte[8];
            if (stream.Read(length) != 8)
            {
                return false;
            }

            long size = System.Buffers.Binary.BinaryPrimitives.ReadInt64LittleEndian(length);
            if (size <= 0 || size > MaxSafetensorsHeader)
            {
                return false;
            }

            var header = new byte[size];
            stream.ReadExactly(header);
            return header.AsSpan().IndexOf(Encoding.ASCII.GetBytes(text)) >= 0;
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            return false;
        }
    }

    /// <summary>
    /// A stable-diffusion.cpp GGUF (a <c>model.diffusion_model.</c> denoiser), or a diffusion transformer (FLUX, SD3: their GGUF
    /// architectures or block names), which the Image page lists so that loading one says whether it is supported.
    /// </summary>
    private static bool GgufHasDiffusionModel(string path)
    {
        try
        {
            using var file = GgufFile.Open(path);
            return file.Architecture is { } architecture && s_diffusionTransformerArchitectures.Contains(architecture)
                || file.Tensors.Any(tensor => tensor.Name.StartsWith(UNetPrefix, StringComparison.Ordinal)
                    || tensor.Name.StartsWith("double_blocks.", StringComparison.Ordinal) || tensor.Name.StartsWith("joint_blocks.", StringComparison.Ordinal)
                    || tensor.Name.EndsWith(QwenImage21TransformerTensor, StringComparison.Ordinal) || tensor.Name.EndsWith(Krea2TransformerTensor, StringComparison.Ordinal));
        }
        catch (Exception error) when (error is IOException or InvalidDataException or NotSupportedException or UnauthorizedAccessException)
        {
            return false;
        }
    }

    /// <summary>The vision kind of a Hugging Face <c>config.json</c> (by its architecture, then its model type), or null.</summary>
    private static string? KindOfConfig(string config)
    {
        if (!File.Exists(config))
        {
            return null;
        }

        try
        {
            using var document = JsonDocument.Parse(File.ReadAllBytes(config));
            var root = document.RootElement;
            string type = root.TryGetProperty("model_type", out var typeElement) ? typeElement.GetString() ?? "" : "";
            string architecture = root.TryGetProperty("architectures", out var list) && list.ValueKind == JsonValueKind.Array && list.GetArrayLength() > 0
                ? list[0].GetString() ?? ""
                : "";
            if (architecture.EndsWith("ForObjectDetection", StringComparison.Ordinal) || type is "detr" or "rt_detr" or "conditional_detr" or "deformable_detr" or "yolos")
            {
                return ModelKinds.ObjectDetection;
            }

            if (type is "sam2" or "sam2_video" or "sam" or "sam_hq")
            {
                return ModelKinds.ImageSegmentation;
            }

            if (type == "vision-encoder-decoder" && root.TryGetProperty("decoder", out var decoder) && decoder.TryGetProperty("model_type", out var decoderType) && decoderType.GetString() == "trocr")
            {
                return ModelKinds.Ocr;
            }

            if (architecture.EndsWith("ForSemanticSegmentation", StringComparison.Ordinal) || architecture == "BiRefNet" || type == "segformer" && !architecture.EndsWith("ForImageClassification", StringComparison.Ordinal))
            {
                return ModelKinds.SemanticSegmentation;
            }

            if (architecture.EndsWith("ForDepthEstimation", StringComparison.Ordinal) || type is "depth_anything" or "dpt")
            {
                return ModelKinds.DepthEstimation;
            }

            if (architecture.EndsWith("ForImageClassification", StringComparison.Ordinal))
            {
                return ModelKinds.ImageClassification;
            }

            // timm checkpoints have an architecture name and a pretrained_cfg instead of a model_type.
            if (type.Length == 0 && root.TryGetProperty("pretrained_cfg", out _) && root.TryGetProperty("architecture", out _))
            {
                return ModelKinds.ImageClassification;
            }

            // Encoders without a classifier head: CLIP and SigLIP (image and text towers), DINOv2 and plain ViT trunks.
            return type is "clip" or "siglip" or "siglip2" or "dinov2" or "dino" or "vit" or "deit" or "beit" or "owlvit" ? ModelKinds.ImageEmbedding : null;
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or JsonException)
        {
            return null;
        }
    }
}
